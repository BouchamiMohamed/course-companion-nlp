"""Live feature checks; structural validation is not a human correctness grade."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from study_assistant.corpus import chunk_document
from study_assistant.llm import Ollama
from study_assistant.presentation import readable_citations
from study_assistant.retrieval import VectorIndex
from study_assistant.services import answer, make_quiz, summarize, validate_citations, validate_quiz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', default='data/sample_nlp_course.txt')
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--skip-summaries', action='store_true')
    parser.add_argument('--output', default='output/acceptance.json')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    course = Path(args.course)
    data = course.read_bytes()
    passages = chunk_document(course.name, data)
    model = Ollama()
    model.check_connection(include_embeddings=args.semantic)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'course': course.name, 'sha256': hashlib.sha256(data).hexdigest(),
              'passages': len(passages), 'model': model.model,
              'retrieval': 'semantic' if args.semantic else 'tfidf', 'cases': [],
              'limits': 'Structural and availability checks only. Review pedagogical and factual correctness separately.'}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save(case):
        report['cases'].append(case)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{case["name"]}: {case["status"]} ({case["seconds"]}s)', flush=True)

    for level, count in [('Easy', 3)] * args.repeats + [('Medium', 3), ('Hard', 3), ('Easy', 8)]:
        started = time.perf_counter()
        quiz = make_quiz(passages, level, count, model)
        validate_quiz({'questions': quiz}, passages, count)
        generated = sum(q['generation'] == 'llm' for q in quiz)
        save({'name': f'quiz_{level}_{count}_{len(report["cases"]) + 1}',
              'status': 'llm_validated' if generated == count else 'practice_fallback',
              'llm_questions': generated, 'requested': count, 'questions': quiz,
              'seconds': round(time.perf_counter() - started, 1)})

    if not args.skip_summaries:
        original = model.generate
        intermediate_cache = {}

        def cached(instruction, context, schema=None):
            # Reuse identical map/reduce operations across the three lengths.
            # Every distinct final-length instruction is still a real model call.
            is_intermediate = instruction.startswith(('Extract key concepts', 'Merge these course notes'))
            key = hashlib.sha256(json.dumps([instruction, context, schema], sort_keys=True).encode()).hexdigest()
            if is_intermediate and key in intermediate_cache:
                return intermediate_cache[key]
            result = original(instruction, context, schema)
            if is_intermediate:
                intermediate_cache[key] = result
            return result

        model.generate = cached
        for length in ('Short', 'Medium', 'Detailed'):
            started = time.perf_counter()
            text = summarize(passages, length, model)
            validate_citations(text, passages)
            save({'name': f'summary_{length}',
                  'status': 'extractive_fallback' if 'Extractive fallback' in text else 'llm_validated',
                  'text': readable_citations(text, passages),
                  'seconds': round(time.perf_counter() - started, 1)})
        model.generate = original

    started = time.perf_counter()
    index = VectorIndex(passages, model.embed if args.semantic else None)
    save({'name': 'vector_index', 'status': 'ready', 'vectors': len(passages),
          'seconds': round(time.perf_counter() - started, 1)})
    questions = ['Quels sont les tablespaces système ?', 'Comment supprimer un tablespace ?',
                 'Qui a remporté la Coupe du monde de football ?'] if course.suffix == '.pdf' else [
                     'What is tokenization?', 'How does RAG work?', 'Who won the football World Cup?']
    for question in questions:
        started = time.perf_counter()
        text, hits = answer(question, index, model)
        if hits:
            validate_citations(text, [p for p, _ in hits])
        save({'name': 'chat', 'question': question,
              'status': 'evidence_fallback' if 'Evidence fallback' in text else 'completed',
              'text': readable_citations(text, passages), 'sources': len(hits),
              'seconds': round(time.perf_counter() - started, 1)})
    print(f'Report: {destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
