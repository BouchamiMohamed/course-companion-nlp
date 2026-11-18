"""UI smoke checks using Streamlit's supported headless test runner."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from study_assistant.llm import ModelError

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class AppTests(unittest.TestCase):
    def test_embedding_failure_keeps_library_usable(self):
        app = AppTest.from_file(APP, default_timeout=20).run()
        with patch('study_assistant.llm.Ollama.embed', side_effect=ModelError('Offline')):
            app.sidebar.selectbox[1].select('Semantic embeddings (Ollama)').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any('Using lexical retrieval' in x.value for x in app.warning))
        self.assertTrue(app.session_state['index'].passages)

    def test_quiz_survives_model_failure_and_failed_regeneration(self):
        app = AppTest.from_file(APP, default_timeout=20).run()
        app.sidebar.selectbox[0].select('Ollama LLM').run()
        with patch('study_assistant.llm.Ollama.generate', side_effect=ModelError('Server unavailable')):
            next(b for b in app.button if b.label == 'Generate quiz').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        quiz = app.session_state['quiz']
        self.assertEqual(len(quiz), 4)
        self.assertTrue(any('source-completion practice' in x.value for x in app.info))
        next(r for r in app.radio if r.label.startswith('1. ')).set_value(0).run()
        next(b for b in app.button if b.label == 'Check answers').click().run()
        self.assertIsNotNone(app.session_state['quiz_answers'])
        with patch('study_assistant.services.make_quiz', side_effect=ValueError('Not enough material')):
            next(b for b in app.button if b.label == 'Generate quiz').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['quiz'], quiz)
        self.assertIsNotNone(app.session_state['quiz_answers'])

    def test_llm_summary_renders_validated_references(self):
        def generated(client, instruction, context, schema=None):
            refs = schema['properties']['notes']['items']['properties']['source_ids']['items']['enum']
            text = 'A concise course summary. ' + 'This passage explains a key course concept in clear terms. ' * 8
            return json.dumps({'notes': [{'heading': 'Key ideas', 'text': text, 'source_ids': [refs[0]]}]})
        app = AppTest.from_file(APP, default_timeout=20).run()
        app.sidebar.selectbox[0].select('Ollama LLM').run()
        with patch('study_assistant.llm.Ollama.generate', new=generated):
            next(b for b in app.button if b.label == 'Create summary').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn('A concise course summary.', app.session_state['summary'])
        self.assertIn('[', app.session_state['summary'])
        visible = '\n'.join(element.value for element in app.markdown)
        self.assertIn('Source:', visible)
        self.assertIn('### Key ideas', visible)
        self.assertIn('PDF p. 1', visible)
        self.assertNotIn(app.session_state['index'].passages[0].id, visible)

    def test_sample_chat_summary_and_clear(self):
        app = AppTest.from_file(APP, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, '1')
        self.assertTrue(next(b for b in app.button if b.label == 'Generate quiz').disabled)
        app.chat_input[0].set_value('What is tokenization?').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state['messages']), 2)
        self.assertIn('Offline preview', app.session_state['messages'][1]['content'])
        next(b for b in app.button if b.label == 'Create summary').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Offline extractive preview', app.session_state['summary'])
        next(b for b in app.button if b.label == 'Clear conversation').click().run()
        self.assertEqual(app.session_state['messages'], [])

    def test_empty_library_and_missing_ollama(self):
        app = AppTest.from_file(APP, default_timeout=20).run()
        app.checkbox[0].uncheck().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Upload a PDF' in x.value for x in app.info))
        app.checkbox[0].check().run()
        app.sidebar.selectbox[0].select('Ollama LLM').run()
        with patch('study_assistant.llm.urlopen', side_effect=URLError('test server unavailable')):
            app.chat_input[0].set_value('What is tokenization?').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn('Evidence fallback', app.session_state['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
