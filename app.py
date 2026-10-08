import streamlit as st
from rag_pipeline import (load_all_pdfs, get_embeddings, build_faiss_index,
                          search, generate_answer, model)

@st.cache_resource
def setup():
    chunks, sources = load_all_pdfs("data")
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)
    return chunks, sources, index

chunks, sources, index = setup()

st.title("RAG Study Assistant")
st.caption(f"{len(chunks)} chunks loaded from the data folder")

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