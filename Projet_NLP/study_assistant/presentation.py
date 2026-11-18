"""Human-readable source labels; internal passage IDs remain internal."""
from collections import defaultdict
import re


def source_label(passages):
    grouped = defaultdict(set)
    for p in passages:
        grouped[p.source].add(p.page)
    labels = []
    for name, page_set in grouped.items():
        pages = sorted(page_set)
        spans = []
        start = previous = pages[0]
        for page in pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            spans.append(str(start) if start == previous else f'{start}–{previous}')
            start = previous = page
        spans.append(str(start) if start == previous else f'{start}–{previous}')
        labels.append(f'{name} · PDF p. {", ".join(spans)}')
    return '; '.join(labels)


def readable_citations(text, passages):
    by_id = {p.id: p for p in passages}

    def replace(match):
        ids = re.findall(r'\[([^\[\]\n]+)\]', match.group())
        if not ids or any(ref not in by_id for ref in ids):
            return match.group()
        label = source_label([by_id[ref] for ref in ids])
        # Escape Markdown punctuation in filenames, keeping labels as plain prose.
        label = re.sub(r'([\\*_\[\]`])', r'\\\1', label)
        return f'\n\n*Source: {label}*\n\n'

    result = re.sub(r'(?:\[[^\[\]\n]+\][ \t]*)+', replace, text)
    return re.sub(r'\n{3,}', '\n\n', result).strip()
