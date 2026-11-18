import unittest
import json
from urllib.error import HTTPError, URLError
from io import BytesIO
from unittest.mock import patch

from pypdf import PdfWriter

from study_assistant.corpus import Passage, chunk_document, extract_pages
from study_assistant.llm import ModelError, Ollama
from study_assistant.retrieval import VectorIndex
from study_assistant.services import answer, make_quiz, summarize, validate_citations, validate_quiz


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.passages = [Passage('a', 'course.txt', 1, 'Tokenization splits text into words and subwords.'),
                         Passage('b', 'course.txt', 2, 'Cosine similarity measures the angle between vectors.')]

    def test_overlap_and_coverage(self):
        words = [f'word{i}' for i in range(75)]
        chunks = chunk_document('x.txt', ' '.join(words).encode(), 30, 5)
        self.assertEqual(chunks[0].text.split()[-5:], chunks[1].text.split()[:5])
        self.assertEqual(set(' '.join(p.text for p in chunks).split()), set(words))
        self.assertEqual(len({p.id for p in chunks}), len(chunks))

    def test_document_ids_distinguish_same_content_different_names(self):
        self.assertNotEqual(chunk_document('a.txt', b'hello')[0].id,
                            chunk_document('b.txt', b'hello')[0].id)

    def test_numbered_section_boundaries_do_not_mix_topics(self):
        text = '1. First topic\n' + 'alpha ' * 30 + '\n2. Second topic\n' + 'beta ' * 30
        chunks = chunk_document('sections.txt', text.encode(), 100, 20)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(not ('alpha' in p.text and 'beta' in p.text) for p in chunks))
        self.assertEqual(len({p.id for p in chunks}), 2)

    def test_empty_and_scanned_documents_rejected(self):
        with self.assertRaises(ValueError):
            extract_pages('empty.txt', b'  ')
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buffer = BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, 'OCR'):
            extract_pages('scan.pdf', buffer.getvalue())

    def test_lexical_ranking_and_unknown_question(self):
        index = VectorIndex(self.passages)
        self.assertEqual(index.search('tokenization')[0][0].id, 'a')
        response, hits = answer('volcanoes', index)
        self.assertEqual(hits, [])
        self.assertIn('not find enough evidence', response)

    def test_semantic_ranking(self):
        def embed(texts):
            return [[1, 0] if ('Tokenization' in t or 'segment' in t) else [0, 1] for t in texts]
        index = VectorIndex(self.passages, embed)
        self.assertEqual(index.search('segment language')[0][0].id, 'a')

    def test_bad_embeddings_rejected(self):
        with self.assertRaises(ValueError):
            VectorIndex(self.passages, lambda _: [[float('nan')]])

    def test_citation_validation(self):
        self.assertEqual(validate_citations('Definition [a]', self.passages), 'Definition [a]')
        for text in ['No reference', 'Invented source [x]', 'Mixed [a] [x]']:
            with self.assertRaises(ModelError):
                validate_citations(text, self.passages)

    def test_quiz_evidence_and_answer_validation(self):
        q = {'question': 'What does tokenization do?', 'options': ['Split text', 'Train', 'Sort', 'Encrypt'],
             'answer': 0, 'explanation': 'It splits text.', 'source_id': 'a',
             'quote': 'Tokenization splits text'}
        self.assertEqual(validate_quiz({'questions': [q]}, self.passages, 1), [q])
        for bad in [{'answer': True}, {'quote': 'A fabricated quotation'}, {'options': ['x'] * 4},
                    {'source_id': 'missing'}]:
            with self.assertRaises(ModelError):
                validate_quiz({'questions': [{**q, **bad}]}, self.passages, 1)

    def test_quiz_repairs_bad_output_once(self):
        from unittest.mock import Mock
        q = {'question': 'What does tokenization do?', 'options': ['Split text', 'Train', 'Sort', 'Encrypt'],
             'answer': 0, 'explanation': 'It splits text.', 'source_id': 'a',
             'evidence_id': 'E1'}
        model = Mock()
        model.generate.side_effect = [json.dumps({'questions': [{**q, 'options': ['x'] * 4}]}),
                                      json.dumps({'questions': [q]})]
        result = make_quiz(self.passages, 'Easy', 1, model)
        self.assertEqual(result[0]['generation'], 'llm')
        self.assertEqual(result[0]['quote'], self.passages[0].text)
        self.assertEqual(model.generate.call_count, 2)
        self.assertIn('distinct options', model.generate.call_args.args[0])
        model.generate.side_effect = ['invalid json', 'invalid json']
        fallback = make_quiz(self.passages, 'Easy', 1, model)
        self.assertEqual(fallback[0]['generation'], 'extractive')
        validate_quiz({'questions': fallback}, self.passages, 1)

    def test_long_summary_reads_every_passage(self):
        class FakeModel:
            def __init__(self):
                self.contexts = []

            def generate(self, instruction, context, schema=None):
                self.contexts.append(context)
                refs = schema['properties']['notes']['items']['properties']['source_ids']['items']['enum']
                return json.dumps({'notes': [{'heading': 'Topic', 'text': 'Study note covers concepts', 'source_ids': [refs[0]]}]})
        passages = [Passage(f'source{i}', 'course', i, f'UniqueConcept{i}') for i in range(31)]
        model = FakeModel()
        summarize(passages, 'Short', model)
        combined = '\n'.join(model.contexts)
        for i in range(31):
            self.assertIn(f'UniqueConcept{i}', combined)

    def test_summary_rejects_and_repairs_unknown_source(self):
        from unittest.mock import Mock
        model = Mock()
        model.generate.side_effect = [
            json.dumps({'notes': [{'text': 'Fact', 'source_ids': ['invented']}]}),
            json.dumps({'notes': [{'text': 'Corrected fact', 'source_ids': ['S1']}]}),
            json.dumps({'notes': [{'heading': 'Key idea', 'text': 'Summary fact', 'source_ids': ['S1']}]})]
        self.assertEqual(summarize(self.passages, 'Short', model), '### Key idea\n\nSummary fact [a]')
        self.assertEqual(model.generate.call_count, 3)
        # The merge can reference only sources actually retained by its input notes.
        schema = model.generate.call_args.args[2]
        self.assertEqual(schema['properties']['notes']['items']['properties']['source_ids']['items']['enum'], ['S1'])

    def test_summary_never_silently_accepts_missing_citations(self):
        from unittest.mock import Mock
        model = Mock()
        model.generate.return_value = json.dumps({'notes': [{'text': 'Unsupported', 'source_ids': []}]})
        result = summarize(self.passages, 'Short', model)
        self.assertIn('Extractive fallback', result)
        validate_citations(result, self.passages)
        self.assertEqual(model.generate.call_count, 2)

    def test_french_course_language_survives_summary_reduction(self):
        from unittest.mock import Mock
        model = Mock()
        model.generate.return_value = json.dumps({'notes': [{'heading': 'Notion', 'text': 'Résumé du cours.', 'source_ids': ['S1']}]})
        passages = [Passage('fr', 'cours.pdf', 1,
                    'Les tablespaces sont des unités de stockage. Une table est dans un tablespace. '
                    'Le stockage et les fichiers sont gérés avec des commandes pour la base.')]
        for length in ('Short', 'Medium', 'Detailed'):
            summarize(passages, length, model)
        for call in model.generate.call_args_list:
            self.assertIn('Write entirely in French', call.args[0])

    def test_ollama_contract(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"message":{"content":"Answer [a]"}}'
        with patch('study_assistant.llm.urlopen', return_value=Response()) as mock:
            self.assertEqual(Ollama().generate('Question', 'Evidence'), 'Answer [a]')
            self.assertTrue(mock.call_args.args[0].full_url.endswith('/api/chat'))

    def test_connection_error_is_actionable(self):
        with patch('study_assistant.llm.urlopen', side_effect=URLError('refused')):
            with self.assertRaisesRegex(ModelError, 'ollama serve'):
                Ollama().generate('Question', 'Evidence')

    def test_missing_model_error_names_pull_command(self):
        error = HTTPError('http://localhost:11434/api/chat', 404, 'Not Found', {},
                          BytesIO(b'{"error":"model qwen2.5:3b not found"}'))
        with patch('study_assistant.llm.urlopen', side_effect=error):
            with self.assertRaisesRegex(ModelError, 'ollama pull qwen2.5:3b'):
                Ollama().generate('Question', 'Evidence')

    def test_slow_model_error(self):
        with patch('study_assistant.llm.urlopen', side_effect=TimeoutError()):
            with self.assertRaisesRegex(ModelError, '180 seconds'):
                Ollama().generate('Question', 'Evidence')

    def test_connection_check_requires_selected_models(self):
        class Response(BytesIO):
            pass
        with patch('study_assistant.llm.urlopen', return_value=Response(
                json.dumps({'models': [{'name': 'qwen2.5:3b'}]}).encode())):
            self.assertIn('qwen2.5:3b', Ollama().check_connection())
        with patch('study_assistant.llm.urlopen', return_value=Response(
                json.dumps({'models': [{'name': 'qwen2.5:3b'}]}).encode())):
            with self.assertRaisesRegex(ModelError, 'ollama pull nomic-embed-text'):
                Ollama().check_connection(include_embeddings=True)


if __name__ == '__main__':
    unittest.main()
