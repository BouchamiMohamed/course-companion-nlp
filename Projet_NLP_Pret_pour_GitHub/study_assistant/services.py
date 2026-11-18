import json
import re
import logging
import hashlib
import random

from .llm import ModelError


def evidence(passages):
    return '\n\n'.join(f'[{p.id}] {p.source}, page {p.page}\n{p.text}' for p in passages)


def validate_citations(text, passages):
    allowed = {p.id for p in passages}
    candidates = re.findall(r'\[([^\[\]\n]+)\]', text)
    cited = {ref for ref in candidates if ref in allowed or re.fullmatch(r'[a-z0-9][a-z0-9_-]*', ref)}
    if not cited or not cited.issubset(allowed):
        raise ModelError('The answer has missing or invalid source references. Please try again.')
    return text


def answer(question, index, model=None, k=4, minimum_score=0.08):
    hits = [(p, score) for p, score in index.search(question, k) if score >= minimum_score]
    if not hits:
        return 'I could not find enough evidence in the uploaded course documents.', []
    passages = [p for p, _ in hits]
    if model is None:
        return ('Offline preview — retrieved course passages:\n\n' +
                '\n\n'.join(f'{p.text} [{p.id}]' for p in passages)), hits
    try:
        text, _ = _cited_notes(model,
            f'Answer this question directly: {question}. '
            'If the evidence does not answer it, explicitly say the evidence is insufficient. '
            'Do not infer general rules from numerical examples. Never use external knowledge.' +
            _language_instruction(passages), evidence(passages), [p.id for p in passages], 4)
        return text, hits
    except ModelError as exc:
        logging.getLogger(__name__).warning('Answer generation unavailable; returning retrieved evidence: %s', exc)
        return ('**Evidence fallback — the model could not generate a validated answer.** '
                'These are retrieved excerpts, not a generated answer:\n\n' +
                '\n\n'.join(f'{p.text} [{p.id}]' for p in passages)), hits


def _cited_notes(model, instruction, context, source_ids, max_blocks, headings=False, min_words=0):
    """Keep source IDs separate from prose, then render citations in Python."""
    allowed = sorted(set(source_ids))
    aliases = {f'S{i + 1}': ref for i, ref in enumerate(allowed)}
    for alias, ref in aliases.items():
        context = context.replace(f'[{ref}]', f'[{alias}]')
    schema = {
        'type': 'object', 'properties': {'notes': {
            'type': 'array', 'minItems': 1, 'maxItems': max_blocks,
            'items': {'type': 'object', 'properties': {
                'text': {'type': 'string', 'minLength': 1},
                'source_ids': {'type': 'array', 'minItems': 1, 'maxItems': min(6, len(allowed)),
                               'items': {'type': 'string', 'enum': list(aliases)}}},
                'required': ['text', 'source_ids'], 'additionalProperties': False}}},
        'required': ['notes'], 'additionalProperties': False,
    }
    if headings:
        item = schema['properties']['notes']['items']
        item['properties']['heading'] = {'type': 'string', 'minLength': 1}
        item['required'].append('heading')
    prompt = (instruction + '\nReturn notes with text and source_ids. Each note needs supporting source_ids '
              'selected exactly from the supplied evidence. Do not put citation markers inside text; '
              'square brackets that belong to SQL syntax or mathematical notation are allowed. '
              'the application will attach citations. Use only facts supported by the supplied evidence.')
    if headings:
        prompt += ' Give each note a short descriptive heading in the heading field; do not repeat it in text.'
    for attempt in range(2):
        raw = model.generate(prompt, context, schema)
        try:
            blocks = json.loads(raw)['notes']
            if not isinstance(blocks, list) or not 1 <= len(blocks) <= max_blocks:
                raise ValueError('Return a nonempty list within the requested note limit.')
            for block in blocks:
                if not isinstance(block, dict):
                    raise ValueError('Each note must be an object.')
                if headings and (not isinstance(block.get('heading'), str) or not block['heading'].strip()
                                 or '[' in block['heading'] or ']' in block['heading']):
                    raise ValueError('Every note needs a short descriptive heading without citation markers.')
                if not isinstance(block.get('text'), str) or not block['text'].strip():
                    raise ValueError('Each text must be nonempty prose.')
                block['text'] = re.sub(r'\[(?:S\d+|[0-9a-f]{10}-\d+-\d+)\]', '', block['text']).strip()
                if not block['text']:
                    raise ValueError('Each note needs actual content, not just citation markers.')
                refs = block.get('source_ids')
                if (not isinstance(refs, list) or not 1 <= len(refs) <= min(6, len(allowed)) or
                        not all(isinstance(ref, str) and ref in aliases for ref in refs)):
                    raise ValueError('Every note needs source_ids selected from its supplied evidence.')
                block['source_ids'] = [aliases[ref] for ref in refs]
            unique = {}
            for block in blocks:
                key = re.sub(r'\W+', ' ', block['text']).strip().casefold()
                if key not in unique:
                    unique[key] = dict(block, source_ids=list(dict.fromkeys(block['source_ids'])))
                else:
                    unique[key]['source_ids'] = list(dict.fromkeys(
                        unique[key]['source_ids'] + block['source_ids']))
            blocks = list(unique.values())
            if sum(len(b['text'].split()) for b in blocks) < min_words:
                raise ValueError(f'Provide at least {min_words} words of distinct content covering different course topics. '
                                 'Repeated paragraphs do not count toward this minimum.')
            rendered = '\n\n'.join((f'### {b["heading"].strip()}\n\n' if headings else '') +
                b['text'].strip() + ' ' + ' '.join(
                f'[{ref}]' for ref in dict.fromkeys(b['source_ids'])) for b in blocks)
            return rendered, set(ref for b in blocks for ref in b['source_ids'])
        except (ValueError, KeyError, TypeError) as exc:
            if attempt:
                raise ModelError(f'Could not generate a summary with valid source references: {exc}') from exc
            prompt += f'\nThe previous output was invalid: {exc}. Return a corrected complete response.'


def _language_instruction(passages):
    words = re.findall(r'[^\W_]+', ' '.join(p.text for p in passages).casefold())
    markers = {
        'French': set('le la les des du une et dans est sont pour avec sur aux ces'.split()),
        'English': set('the and of to in is are with for from that this these'.split()),
        'Arabic': set('في من على إلى عن هذا هذه التي الذي يتم هو هي'.split()),
        'Portuguese': set('os uma são dos das para não pelo pela aos'.split()),
        'Spanish': set('los las una son para del por con que como'.split()),
    }
    scores = {name: sum(word in terms for word in words) for name, terms in markers.items()}
    language = max(scores, key=scores.get)
    return (f' Write entirely in {language}, the original course language; do not translate.'
            if scores[language] >= 5 else ' Keep the original course language; do not translate.')


def _summarize(passages, length, model=None):
    targets = {'Short': 100, 'Medium': 250, 'Detailed': 500}
    target = targets[length]
    if model is None:
        count = min(len(passages), {'Short': 2, 'Medium': 4, 'Detailed': 8}[length])
        selected = [passages[i * len(passages) // count] for i in range(count)]
        return 'Offline extractive preview:\n\n' + '\n\n'.join(
            ' '.join(p.text.split()[:max(15, target // count)]) + f' [{p.id}]' for p in selected)
    language_instruction = _language_instruction(passages)
    # Map/reduce reads every passage, rather than summarizing only the first context window.
    notes = []
    for start in range(0, len(passages), 5):
        batch = passages[start:start + 5]
        notes.append(_cited_notes(model,
            'Extract key concepts, definitions and relationships in at most 180 words total.' + language_instruction,
            evidence(batch), [p.id for p in batch], max_blocks=4))
    # Hierarchical reduction keeps long courses within the model context budget.
    while len(notes) > 5:
        reduced = []
        for i in range(0, len(notes), 5):
            group = notes[i:i + 5]
            reduced.append(_cited_notes(model,
                'Merge these course notes into at most 250 words total.' + language_instruction,
                '\n\n'.join(text for text, _ in group),
                set().union(*(refs for _, refs in group)), max_blocks=5))
        notes = reduced
    result, _ = _cited_notes(model,
        f'Create a {length.lower()} study summary of about {target} words total. '
        'Cover the main topics across the beginning, middle and end of the course. '
        'Make this easy for a student to revise: start each note with a short bold topic heading, '
        'then use short sentences or bullet points. Explain one idea per sentence. '
        'Define technical terms before using them. Preserve distinctions and intermediate levels '
        'in hierarchies. Do not repeat information or copy page headers, section numbers or contact details. '
        'Do not add facts or examples absent from the course.' + language_instruction,
        '\n\n'.join(text for text, _ in notes),
        set().union(*(refs for _, refs in notes)),
        max_blocks={'Short': 3, 'Medium': 6, 'Detailed': 10}[length], headings=True,
        min_words=min(target // 2, sum(len(p.text.split()) for p in passages) // 10))
    return result


def summarize(passages, length, model=None):
    if not passages:
        raise ValueError('Upload readable course text before requesting a summary.')
    if length not in {'Short', 'Medium', 'Detailed'}:
        raise ValueError('Choose Short, Medium, or Detailed.')
    try:
        return _summarize(passages, length, model)
    except ModelError:
        logging.getLogger(__name__).warning('Summary generation unavailable; returning extractive fallback.')
        return ('**Extractive fallback — model generation did not complete successfully.**\n\n' +
                _summarize(passages, length, None).replace('Offline extractive preview:\n\n', '', 1))


def validate_quiz(payload, passages, count):
    try:
        questions = payload['questions']
        by_id = {p.id: p for p in passages}
        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError(f'Return exactly {count} questions.')
        seen = set()
        for q in questions:
            if not isinstance(q, dict):
                raise ValueError('Each question must be a JSON object.')
            for field in ('question', 'explanation', 'source_id', 'quote'):
                if not isinstance(q[field], str) or not q[field].strip():
                    raise ValueError(f'Question field {field} must be a nonempty string.')
            opts = q['options']
            if (not isinstance(opts, list) or len(opts) != 4 or
                    not all(isinstance(o, str) and o.strip() for o in opts) or
                    len({o.strip().casefold() for o in opts}) != 4 or
                    type(q['answer']) is not int or not 0 <= q['answer'] < 4):
                raise ValueError('Each question needs four nonempty, distinct options and an integer answer index from 0 to 3.')
            if q['source_id'] not in by_id or q['question'].casefold() in seen:
                raise ValueError('Questions must be distinct and source_id must match a supplied passage.')
            quote = ' '.join(q['quote'].split())
            if len(quote) < 15 or quote not in ' '.join(by_id[q['source_id']].text.split()):
                raise ValueError('Each quote must be copied verbatim from its source_id passage and contain at least 15 characters.')
            seen.add(q['question'].casefold())
        return questions
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError(f'Quiz validation failed: {exc or "missing required fields"}') from exc


def make_quiz(passages, difficulty, count, model):
    if difficulty not in {'Easy', 'Medium', 'Hard'} or type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Choose a valid difficulty and between 1 and 8 questions.')
    cards = quiz_evidence(passages)
    if len(cards) < count:
        raise ValueError('Not enough distinct readable course evidence for this many questions. Request fewer questions or upload more text.')
    by_id = {card['id']: card for card in cards}
    schema = {
        'type': 'object', 'properties': {'questions': {'type': 'array', 'minItems': 1,
            'maxItems': count, 'items': {'type': 'object', 'properties': {
                'question': {'type': 'string', 'minLength': 1},
                'options': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                            'items': {'type': 'string', 'minLength': 1}},
                'answer': {'type': 'integer', 'minimum': 0, 'maximum': 3},
                'explanation': {'type': 'string', 'minLength': 1},
                'evidence_id': {'type': 'string', 'enum': list(by_id)}},
                'required': ['question', 'options', 'answer', 'explanation', 'evidence_id'],
                'additionalProperties': False}}},
        'required': ['questions'], 'additionalProperties': False}
    levels = {'Easy': 'Test a direct definition or fact.',
              'Medium': 'Test a relationship, distinction or consequence stated in the evidence.',
              'Hard': 'Test careful application of a stated condition. Do not require outside knowledge.'}
    base = (f'Create distinct {difficulty.lower()} multiple-choice questions. {levels[difficulty]} '
            'Each needs four distinct nonempty options, exactly one correct answer (index 0 to 3), '
            'and a concise explanation supported by its selected evidence_id. '
            'Select an evidence_id only; do not copy quotes or invent source metadata. '
            'Evidence IDs are metadata only: never mention E1/E2 or any card ID in the question, options or explanation. '
            'Never turn a numeric example into a general maximum or universal rule. '
            'Do not ask about a fact absent from the selected evidence. '
            'Use a different evidence card for each question.' + _language_instruction(passages))
    accepted, used, seen = [], set(), set()
    # Small local models are more reliable with short outputs than an eight-item
    # JSON object. Allow one repair batch beyond the normal number of batches.
    for attempt in range((count + 2) // 3 + 1):
        available = [c for c in cards if c['id'] not in used]
        if model is None or len(accepted) == count:
            break
        batch_count = min(3, count - len(accepted))
        prompt = base + f'\nReturn exactly {batch_count} new questions.'
        if attempt:
            prompt += ' Continue with new questions. Use distinct options and only the available evidence IDs.'
            prompt += '\nDo not repeat these accepted questions: ' + json.dumps([q['question'] for q in accepted])
        request_schema = json.loads(json.dumps(schema))
        request_schema['properties']['questions'].update(minItems=batch_count, maxItems=batch_count)
        request_schema['properties']['questions']['items']['properties']['evidence_id']['enum'] = [c['id'] for c in available]
        try:
            raw = model.generate(prompt, json.dumps(
                [{'id': c['id'], 'text': c['quote']} for c in available], ensure_ascii=False), request_schema)
            items = json.loads(raw).get('questions', [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get('evidence_id'), str):
                    continue
                card = by_id.get(item['evidence_id'])
                if card is None or card['id'] in used:
                    continue
                candidate = {key: item.get(key) for key in ('question', 'options', 'answer', 'explanation')}
                if isinstance(candidate['explanation'], str):
                    candidate['explanation'] = re.sub(r'\bE\d+\b',
                        'du cours' if 'French' in _language_instruction(passages) else 'from the course',
                        candidate['explanation'])
                # The app owns the pair: no generated quote/source combination is trusted.
                candidate.update(source_id=card['source_id'], quote=card['quote'], generation='llm')
                try:
                    validate_quiz({'questions': [candidate]}, passages, 1)
                except ModelError:
                    continue
                normalized = candidate['question'].strip().casefold()
                if normalized in seen:
                    continue
                accepted.append(candidate)
                seen.add(normalized)
                used.add(card['id'])
                if len(accepted) == count:
                    break
        except ModelError:
            logging.getLogger(__name__).warning('Quiz model unavailable; using evidence practice fallback.')
            break
        except (ValueError, AttributeError, TypeError):
            continue
    if len(accepted) < count:
        accepted.extend(_practice_questions(cards, passages, difficulty, count - len(accepted), used, seen))
    return validate_quiz({'questions': accepted}, passages, count)


def quiz_evidence(passages):
    """Build atomic cards whose source/quote relationship cannot be generated incorrectly."""
    cards = []
    seen = set()
    for p in passages:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p.text)
                     if 50 <= len(s.strip()) <= 300]
        if not sentences:
            sentences = [' '.join(p.text.split()[:45])]
        limit = min(3, len(sentences))
        for i in range(limit):
            quote = sentences[i * len(sentences) // limit]
            if len(quote) >= 15 and quote.casefold() not in seen:
                cards.append({'source_id': p.id, 'quote': quote})
                seen.add(quote.casefold())
    if len(cards) > 32:
        cards = [cards[i * len(cards) // 32] for i in range(32)]
    return [dict(card, id=f'E{i + 1}') for i, card in enumerate(cards)]


def _practice_questions(cards, passages, difficulty, count, used, seen):
    """Literal source completion exercises, explicitly labeled as a fallback."""
    french = 'French' in _language_instruction(passages)
    stop = set(('this that with from have will into when where which their these those '
                'dans pour avec cette cet sont plus une des les aux est sur par tout '
                'doit peut comme alors entre ainsi autre dont chaque').split())
    width = {'Easy': 1, 'Medium': 2, 'Hard': 3}[difficulty]

    def phrases(text):
        matches = list(re.finditer(r'[^\W\d]\w{2,}', text, re.UNICODE))
        found = []
        for i, match in enumerate(matches):
            if match.group().casefold() in stop or i + width > len(matches):
                continue
            end = matches[i + width - 1].end()
            phrase = text[match.start():end]
            if len(phrase) <= 65:
                found.append(phrase)
        return sorted(set(found), key=lambda value: (not value.isupper(), -len(value), value))

    pool = list(dict.fromkeys(word for card in cards for word in phrases(card['quote'])))
    results = []
    for card in cards:
        if card['id'] in used:
            continue
        quote = card['quote']
        for correct in phrases(quote):
            alternatives = [word for word in pool if word.casefold() not in quote.casefold()]
            alternatives = list({word.casefold(): word for word in alternatives}.values())
            if len(alternatives) < 3:
                continue
            seed = int(hashlib.sha256((quote + difficulty).encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            options = [correct] + rng.sample(alternatives, 3)
            rng.shuffle(options)
            masked = re.sub(re.escape(correct), '____', quote, flags=re.IGNORECASE)
            question = ("Complétez exactement cet extrait du cours :\n\n" if french else
                        'Complete this course excerpt exactly:\n\n') + masked
            if question.casefold() in seen:
                continue
            explanation = (f'Le passage original utilise « {correct} ». Il s’agit d’un exercice de rappel fondé sur le texte.'
                           if french else f'The original passage uses “{correct}”. This is a source-based recall exercise.')
            results.append(dict(question=question, options=options, answer=options.index(correct),
                                explanation=explanation, source_id=card['source_id'], quote=quote,
                                generation='extractive'))
            used.add(card['id'])
            seen.add(question.casefold())
            break
        if len(results) == count:
            return results
    raise ValueError('Not enough distinct course material to create that many reliable exercises. Request fewer questions or add documents.')
