import unittest

from study_assistant.corpus import Passage, clean_pdf_margins
from study_assistant.presentation import readable_citations, source_label


class ReadabilityTests(unittest.TestCase):
    def test_references_collapse_chunks_into_document_pages(self):
        passages = [Passage('internal-a', 'Chapter-4.pdf', 1, 'A'),
                    Passage('internal-b', 'Chapter-4.pdf', 1, 'B'),
                    Passage('internal-c', 'Chapter-4.pdf', 2, 'C'),
                    Passage('internal-d', 'other.pdf', 7, 'D')]
        text = readable_citations('Definition. [internal-a] [internal-b] [internal-c]\n\nOther. [internal-d]', passages)
        self.assertNotIn('internal-', text)
        self.assertIn('Chapter-4.pdf · PDF p. 1–2', text)
        self.assertIn('other.pdf · PDF p. 7', text)
        self.assertEqual(text.count('Source:'), 2)
        self.assertEqual(source_label(passages[:2]), 'Chapter-4.pdf · PDF p. 1')

    def test_math_brackets_and_filename_punctuation_are_preserved(self):
        passages = [Passage('source', 'my_course.pdf', 3, 'text')]
        text = readable_citations('Range [0, 1]. [source]', passages)
        self.assertIn('[0, 1]', text)
        self.assertIn(r'my\_course.pdf', text)
        self.assertNotIn('[source]', text)

    def test_margin_cleanup_preserves_body_and_page_metadata(self):
        pages = []
        for i in range(3):
            lines = ['Repeated course title', 'Document 2.0', 'Feedbacks: teacher@example.com', str(35 + i)]
            lines += [f'Unique opening {i}', f'Topic {i}']
            lines += ['Useful repeated body content', 'CREATE TABLESPACE test;', f'Last topic {i}',
                      f'Closing sentence {i}', f'Another closing {i}', str(35 + i)]
            pages.append((i + 1, '\n'.join(lines)))
        cleaned = clean_pdf_margins(pages)
        self.assertEqual([p for p, _ in cleaned], [1, 2, 3])
        for _, text in cleaned:
            self.assertNotIn('teacher@example.com', text)
            self.assertNotIn('Repeated course title', text)
            self.assertIn('Useful repeated body content', text)
            self.assertIn('CREATE TABLESPACE test;', text)
