"""Run retrieval-only evaluation. Generation quality requires separate human review."""
import argparse
import json
from pathlib import Path
import time

from study_assistant.corpus import chunk_document
from study_assistant.llm import Ollama
from study_assistant.retrieval import VectorIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', type=int, default=240)
    parser.add_argument('--overlap', type=int, default=40)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--embedding-model', default='nomic-embed-text')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    corpus = Path('data/sample_nlp_course.txt')
    passages = chunk_document(corpus.name, corpus.read_bytes(), args.chunk_size, args.overlap)
    embed = Ollama(args.ollama_url, embedding_model=args.embedding_model).embed if args.semantic else None
    index = VectorIndex(passages, embed)
    rows = []
    for case in json.loads(Path('data/evaluation.json').read_text(encoding='utf-8')):
        start = time.perf_counter()
        hits = index.search(case['question'], args.top_k)
        relevant = [i + 1 for i, (p, _) in enumerate(hits)
                    if case.get('evidence', '').casefold() in p.text.casefold()] if case['answerable'] else []
        rows.append({**case, 'first_relevant_rank': min(relevant) if relevant else None,
                     'top_similarity': hits[0][1] if hits else 0,
                     'retrieval_ms': round((time.perf_counter() - start) * 1000, 3)})
    answerable = [r for r in rows if r['answerable']]
    result = {'settings': vars(args), 'passages': len(passages),
              'hit_rate_at_k': sum(r['first_relevant_rank'] is not None for r in answerable) / len(answerable),
              'mrr_at_k': sum(1 / r['first_relevant_rank'] if r['first_relevant_rank'] else 0
                             for r in answerable) / len(answerable),
              'note': 'Tiny synthetic retrieval smoke set; not a real-course benchmark. Unanswerable rows report similarity only.',
              'cases': rows}
    Path('output').mkdir(exist_ok=True)
    Path('output/evaluation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
