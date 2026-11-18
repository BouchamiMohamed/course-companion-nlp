from hashlib import sha256
import json
from pathlib import Path

import streamlit as st

from study_assistant.corpus import chunk_document
from study_assistant.llm import ModelError, Ollama
from study_assistant.retrieval import VectorIndex
from study_assistant.services import answer, make_quiz, summarize
from study_assistant.presentation import readable_citations, source_label


st.set_page_config(page_title='Course Companion', page_icon='📚', layout='wide')
st.title('📚 Course Companion')
st.caption('Your courses, explained. Study with summaries, practice questions, and answers grounded in your documents.')

with st.sidebar:
    st.header('Your study space')
    mode = st.selectbox('Mode', ['Offline preview', 'Ollama LLM'])
    url = st.text_input('Ollama address', 'http://localhost:11434', disabled=mode == 'Offline preview')
    model_name = st.text_input('Language model', 'qwen2.5:3b', disabled=mode == 'Offline preview')
    retrieval = st.selectbox('Retrieval', ['TF-IDF baseline', 'Semantic embeddings (Ollama)'])
    embedding_name = st.text_input('Embedding model', 'nomic-embed-text',
                                    disabled=retrieval == 'TF-IDF baseline')
    if mode == 'Ollama LLM':
        if st.button('Check Ollama connection'):
            try:
                Ollama(url, model_name, embedding_name).check_connection(retrieval.startswith('Semantic'))
                st.success('Ollama is ready. Required models are installed.')
            except ModelError as exc:
                st.error(str(exc))
        with st.expander('Ollama setup'):
            st.markdown('[Install Ollama for Windows](https://ollama.com/download/windows), then run:')
            st.code(f'ollama pull {model_name}', language='powershell')
            if retrieval.startswith('Semantic'):
                st.code(f'ollama pull {embedding_name}', language='powershell')
            st.caption('Keep Ollama running. If it is not running, use ollama serve in a terminal.')
    with st.expander('Retrieval settings'):
        size = st.slider('Words per passage', 100, 500, 240, 20)
        overlap = st.slider('Overlap in words', 0, 80, 40, 10)
        top_k = st.slider('Retrieved passages', 1, 8, 4)
        threshold = st.slider('Minimum similarity', 0.0, 0.8, 0.08, 0.01)
        st.caption('Similarity is a ranking score, not a probability that an answer is correct.')
    uploaded = st.file_uploader('Upload course documents', type=['pdf', 'txt'], accept_multiple_files=True)
    use_sample = st.checkbox('Include sample NLP course', value=not bool(uploaded))
    st.caption('Documents stay in this session. In Ollama mode, evidence is sent to the configured Ollama address.')

files = [(f.name, f.getvalue()) for f in uploaded]
if use_sample:
    files.append(('sample_nlp_course.txt', Path('data/sample_nlp_course.txt').read_bytes()))
if not files:
    st.info('Upload a PDF or TXT course, or enable the sample NLP course to get started.')
    st.stop()

client = Ollama(url, model_name, embedding_name)
model = client if mode == 'Ollama LLM' else None
signature = sha256(repr([(n, sha256(b).hexdigest()) for n, b in files]).encode() +
                   repr((size, overlap, retrieval, url, embedding_name)).encode()).hexdigest()
if st.session_state.get('corpus_signature') != signature:
    try:
        with st.spinner('Preparing your course library…'):
            passages = []
            seen = set()
            for name, data in files:
                identity = (name, sha256(data).hexdigest())
                if identity not in seen:
                    passages.extend(chunk_document(name, data, size, overlap))
                    seen.add(identity)
            if len(passages) > 1500:
                raise ValueError('This prototype supports up to 1,500 passages. Upload a smaller course collection.')
            retrieval_notice = None
            try:
                index = VectorIndex(passages, client.embed if retrieval.startswith('Semantic') else None)
            except (ModelError, ValueError):
                if not retrieval.startswith('Semantic'):
                    raise
                index = VectorIndex(passages)
                retrieval_notice = ('Semantic embeddings are unavailable. Using lexical retrieval for this library. '
                                    'After restoring Ollama, change the retrieval setting to rebuild the index.')
        st.session_state.update(corpus_signature=signature, index=index, messages=[], summary=None, quiz=None, quiz_answers=None)
        st.session_state.retrieval_notice = retrieval_notice
    except Exception as exc:
        st.error(f'Could not prepare the documents: {exc}')
        st.stop()
index = st.session_state.index
if st.session_state.get('retrieval_notice'):
    st.warning(st.session_state.retrieval_notice)
generation_signature = (mode, model_name, top_k, threshold)
if st.session_state.get('generation_signature') != generation_signature:
    st.session_state.update(generation_signature=generation_signature, messages=[], summary=None, quiz=None, quiz_answers=None)

a, b, c = st.columns(3)
a.metric('Documents', len(set(p.source for p in index.passages)))
b.metric('Course passages', len(index.passages))
c.metric('Study mode', 'Local LLM' if model else 'Offline preview')
if not model:
    st.info('Offline preview shows retrieved excerpts and extractive summaries. Select Ollama LLM for generated answers and quizzes.')


def show_sources(passages):
    with st.expander('Read source passages'):
        for p in passages:
            st.markdown(f'**{p.source} · page {p.page}**')
            st.text(p.text)


chat, summaries, quizzes, library = st.tabs(['Ask your course', 'Summaries', 'Practice quiz', 'Library'])
with chat:
    st.subheader('Make sense of your course')
    st.caption('Ask a self-contained question, such as “How does retrieval-augmented generation work?”')
    if st.button('Clear conversation'):
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(readable_citations(message['content'], index.passages))
            if message.get('sources'):
                show_sources(message['sources'])
    question = st.chat_input('Ask a question about your course')
    if question:
        with st.chat_message('user'):
            st.markdown(question)
        st.session_state.messages.append({'role': 'user', 'content': question})
        try:
            with st.spinner('Finding course evidence…'):
                text, hits = answer(question, index, model, top_k, threshold)
            sources = [p for p, _ in hits]
            st.session_state.messages.append({'role': 'assistant', 'content': text, 'sources': sources})
            with st.chat_message('assistant'):
                st.markdown(readable_citations(text, index.passages))
                if sources:
                    show_sources(sources)
        except (ModelError, ValueError) as exc:
            st.error(str(exc))

with summaries:
    st.subheader('Review the key ideas')
    length = st.radio('Summary length', ['Short', 'Medium', 'Detailed'], horizontal=True)
    if st.button('Create summary', type='primary'):
        try:
            with st.spinner('Reading and summarizing your course…'):
                st.session_state.summary = summarize(index.passages, length, model)
        except (ModelError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.get('summary'):
        readable_summary = readable_citations(st.session_state.summary, index.passages)
        st.markdown(readable_summary)
        st.download_button('Download summary', readable_summary, 'course_summary.md')
        show_sources(index.passages)

with quizzes:
    st.subheader('Check your understanding')
    difficulty = st.selectbox('Difficulty', ['Easy', 'Medium', 'Hard'])
    count = st.slider('Number of questions', 1, 8, 4)
    if st.button('Generate quiz', type='primary', disabled=model is None):
        try:
            with st.spinner('Writing questions and checking evidence…'):
                quiz = make_quiz(index.passages, difficulty, count, model)
            st.session_state.quiz = quiz
            st.session_state.quiz_answers = None
            st.session_state.quiz_version = st.session_state.get('quiz_version', 0) + 1
        except (ModelError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.get('quiz'):
        fallback_count = sum(q.get('generation') == 'extractive' for q in st.session_state.quiz)
        if fallback_count:
            st.info(f'{fallback_count} question(s) use source-completion practice because model generation '
                    'was unavailable or invalid. These fallback items test recall; they do not claim '
                    'the requested difficulty level.')
        version = st.session_state.quiz_version
        with st.form(f'quiz_{signature}_{version}'):
            choices = []
            for i, q in enumerate(st.session_state.quiz):
                choices.append(st.radio(f'{i + 1}. {q["question"]}', range(4),
                                        format_func=lambda n, opts=q['options']: opts[n],
                                        index=None, key=f'choice_{signature}_{version}_{i}'))
            submitted = st.form_submit_button('Check answers')
        if submitted:
            st.session_state.quiz_answers = choices
        if st.session_state.get('quiz_answers') is not None:
            choices = st.session_state.quiz_answers
            score = sum(choice == q['answer'] for choice, q in zip(choices, st.session_state.quiz))
            st.success(f'Your score: {score}/{len(choices)}')
            for i, (choice, q) in enumerate(zip(choices, st.session_state.quiz)):
                st.markdown(f'**{i + 1}. {"Correct" if choice == q["answer"] else "Review this one"}**')
                st.write(f'Your answer: {q["options"][choice] if choice is not None else "No answer selected"}')
                st.write(f'Correct answer: {q["options"][q["answer"]]}')
                st.write(q['explanation'])
                matched = [p for p in index.passages if p.id == q['source_id']]
                st.caption(f'Source: {source_label(matched)}')
                with st.expander('Read supporting excerpt'):
                    st.write(q['quote'])
        st.download_button('Download quiz with corrections',
                           json.dumps(st.session_state.quiz, ensure_ascii=False, indent=2),
                           'course_quiz.json', mime='application/json')

with library:
    st.subheader('Your course evidence')
    st.caption('PDF references preserve page numbers. TXT documents are treated as one page.')
    selected = st.selectbox('Document', list(dict.fromkeys(p.source for p in index.passages)))
    show_sources([p for p in index.passages if p.source == selected])
