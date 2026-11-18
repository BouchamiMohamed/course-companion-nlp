from dataclasses import dataclass
from io import BytesIO
import hashlib
import re
from collections import Counter

from pypdf import PdfReader


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    page: int
    text: str


def clean_pdf_margins(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove recurring margin lines without discarding repeated body content."""
    lines_by_page = [[line.strip() for line in text.splitlines() if line.strip()]
                     for _, text in pages]
    counts = Counter()
    for lines in lines_by_page:
        margin = lines[:5] + lines[-3:]
        counts.update(set(re.sub(r'\s+', ' ', line) for line in margin if len(line) <= 180))
    recurring = {line for line, count in counts.items() if count >= max(2, (len(pages) + 1) // 2)}
    cleaned = []
    for (number, _), lines in zip(pages, lines_by_page):
        kept = []
        for i, line in enumerate(lines):
            in_margin = i < 5 or i >= len(lines) - 3
            if in_margin and re.sub(r'\s+', ' ', line) in recurring:
                continue
            if (i < 5 or i >= len(lines) - 2) and re.fullmatch(r'\d{1,4}', line):
                continue
            kept.append(line)
        cleaned.append((number, '\n'.join(kept)))
    return cleaned


def extract_pages(name: str, data: bytes) -> list[tuple[int, str]]:
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Each document must be smaller than 20 MB.")
    if name.lower().endswith('.pdf'):
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Please upload an unencrypted PDF.")
        pages = [(i + 1, page.extract_text() or '') for i, page in enumerate(reader.pages)]
        pages = clean_pdf_margins(pages)
    elif name.lower().endswith('.txt'):
        pages = [(1, data.decode('utf-8-sig'))]
    else:
        raise ValueError("Supported formats: PDF and UTF-8 TXT.")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No readable text found. Scanned PDFs require OCR before upload.")
    return pages


def chunk_document(name: str, data: bytes, size: int = 240, overlap: int = 40) -> list[Passage]:
    if size < 20 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be >= 20 and overlap smaller than chunk size.")
    digest = hashlib.sha256(name.encode() + data).hexdigest()[:10]
    passages = []
    for page, text in extract_pages(name, data):
        # Detect numbered section headings before reflowing PDF line breaks.
        # Overlap remains within the section rather than mixing adjacent topics.
        sections = re.split(r'(?m)(?=^[ \t]*(?:\d+(?:\.\d+)+|\d+\.)[ \t]+\S)', text)
        offset = 0
        for section in sections:
            words = re.sub(r'\s+', ' ', section).strip().split()
            for start in range(0, len(words), size - overlap):
                passages.append(Passage(f'{digest}-{page}-{offset + start}', name, page,
                                        ' '.join(words[start:start + size])))
                if start + size >= len(words):
                    break
            offset += len(words)
    return passages
