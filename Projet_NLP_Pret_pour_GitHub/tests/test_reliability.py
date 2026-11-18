import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from study_assistant.corpus import Passage, chunk_document
from study_assistant.llm import ModelError
from study_assistant.retrieval import VectorIndex
from study_assistant.services import answer, make_quiz, quiz_evidence, validate_quiz, _cited_notes


class ReliabilityTests(unittest.TestCase):
    def test_french_inflection_retrieval_and_query_embedding_failure(self):
        passages = [Passage('drop', 'course.pdf', 7, 'Un tablespace est supprimé via la requête DROP.'),
                    Passage('intro', 'course.pdf', 1, 'Un tablespace contient des données logiques.')]
        def embed(texts):
            return [[0.8, 0.6] if 'supprimé' in t else [1, 0] for t in texts]
        index = VectorIndex(passages, embed)
        self.assertEqual(index.search('Comment supprimer un tablespace ?')[0][0].id, 'drop')
        index.embed = Mock(side_effect=ModelError('Offline'))
        self.assertEqual(index.search('Comment supprimer un tablespace ?')[0][0].id, 'drop')

    def test_repeated_summary_is_not_treated_as_a_detailed_success(self):
        model = Mock()
        note = {'heading': 'Repeated topic', 'text': 'The same short paragraph.', 'source_ids': ['S1']}
        model.generate.return_value = json.dumps({'notes': [note] * 10})
        with self.assertRaisesRegex(ModelError, 'distinct content'):
            _cited_notes(model, 'Detailed summary', 'evidence', ['a'], 10, True, min_words=100)

    def test_sql_optional_syntax_is_not_mistaken_for_an_invalid_citation(self):
        model = Mock()
        model.generate.return_value = json.dumps({'notes': [{
            'text': 'Use DROP TABLESPACE name [INCLUDING CONTENTS]. [S1]', 'source_ids': ['S1']}]})
        text, refs = _cited_notes(model, 'Explain SQL', 'evidence', ['a'], 2)
        self.assertIn('[INCLUDING CONTENTS]', text)
        self.assertNotIn('[S1]', text)
        self.assertEqual(refs, {'a'})
        from study_assistant.services import validate_citations
        validate_citations(text, [Passage('a', 'course.pdf', 1, 'SQL')])

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'data/sample_nlp_course.txt'
        cls.passages = chunk_document(path.name, path.read_bytes())

    def test_evidence_pair_owned_by_app_even_if_model_forges_metadata(self):
        card = quiz_evidence(self.passages)[1]
        model = Mock()
        model.generate.return_value = json.dumps({'questions': [{
            'question': 'Which concept is described?', 'options': ['A', 'B', 'C', 'D'],
            'answer': 0, 'explanation': 'Explanation.', 'evidence_id': card['id'],
            'quote': 'Invented quote', 'source_id': 'wrong source'}]})
        quiz = make_quiz(self.passages, 'Easy', 1, model)
        self.assertEqual(quiz[0]['quote'], card['quote'])
        self.assertEqual(quiz[0]['source_id'], card['source_id'])
        self.assertEqual(quiz[0]['generation'], 'llm')

    def test_server_failure_produces_valid_practice_for_every_count_and_level(self):
        model = Mock()
        model.generate.side_effect = ModelError('Connection lost')
        for level in ('Easy', 'Medium', 'Hard'):
            for count in range(1, 9):
                with self.subTest(level=level, count=count):
                    quiz = make_quiz(self.passages, level, count, model)
                    validate_quiz({'questions': quiz}, self.passages, count)
                    self.assertTrue(all(q['generation'] == 'extractive' for q in quiz))
                    for q in quiz:
                        self.assertIn('____', q['question'])
                        self.assertIn(q['options'][q['answer']], q['quote'])
                        for i, option in enumerate(q['options']):
                            if i != q['answer']:
                                self.assertNotIn(option.casefold(), q['quote'].casefold())

    def test_malformed_and_partial_outputs_keep_working(self):
        invalid = ['broken json', '[]', 'null', '{"questions":null}',
                   '{"questions":[null,5,{"evidence_id":[]}]}']
        for raw in invalid:
            with self.subTest(raw=raw):
                model = Mock()
                model.generate.return_value = raw
                quiz = make_quiz(self.passages, 'Easy', 3, model)
                self.assertEqual(len(quiz), 3)
                self.assertTrue(all(q['generation'] == 'extractive' for q in quiz))
                self.assertEqual(model.generate.call_count, 2)
        model = Mock()
        item = {'question': 'An accepted question?', 'options': ['A', 'B', 'C', 'D'],
                'answer': 1, 'explanation': 'Evidence.', 'evidence_id': 'E1'}
        model.generate.return_value = json.dumps({'questions': [item, item]})
        quiz = make_quiz(self.passages, 'Medium', 3, model)
        self.assertEqual(quiz[0]['question'], item['question'])
        self.assertEqual(sum(q['generation'] == 'llm' for q in quiz), 1)
        self.assertEqual(len(quiz), 3)

    def test_no_fake_success_for_empty_or_insufficient_course(self):
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            make_quiz([], 'Easy', 3, None)
        with self.assertRaises(ValueError):
            make_quiz(self.passages, 'Easy', 9, None)

    def test_chat_model_failure_is_explicit_and_has_real_evidence(self):
        model = Mock()
        model.generate.side_effect = ModelError('Server unavailable')
        text, hits = answer('What is tokenization?', VectorIndex(self.passages), model)
        self.assertIn('Evidence fallback', text)
        self.assertTrue(hits)
        self.assertIn(hits[0][0].id, text)
