import streamlit as st
from rag_pipeline import (load_pdf, chunk_text, get_embeddings,
                          build_faiss_index, search, generate_answer, model)

def build_index(files):
    all_chunks, all_sources = [], []
    for f in files:
        text = load_pdf(f)
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        all_chunks.extend(chunks)
        all_sources.extend([f.name] * len(chunks))
    embeddings = get_embeddings(all_chunks)
    index = build_faiss_index(embeddings)
    return all_chunks, all_sources, index

st.title("RAG Study Assistant")
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf",
                                  accept_multiple_files=True)

if uploaded_files:
    key = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("key") != key:
        with st.spinner("Reading your documents..."):
            st.session_state.data = build_index(uploaded_files)
            st.session_state.key = key
    chunks, sources, index = st.session_state.data
    st.caption(f"{len(chunks)} chunks ready")

    question = st.text_input("Ask a question about your documents")
    if question:
        with st.spinner("Thinking..."):
            results, result_sources = search(question, model, index, chunks, sources, top_k=5)
            answer = generate_answer(question, results)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for i, (chunk, source) in enumerate(zip(results, result_sources)):
            with st.expander(f"Chunk {i+1}: {source}"):
                st.write(chunk)
else:
    st.info("Upload one or more PDFs to get started.")