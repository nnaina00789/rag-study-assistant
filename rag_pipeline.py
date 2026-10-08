import ollama,os
from pypdf import PdfReader
def load_pdf(path):
    reader=PdfReader(path)
    print(f"Number of pages: {len(reader.pages)}")
    full_text=""
    for page in reader.pages:
        full_text+=page.extract_text()
    return full_text
def chunk_text(text,chunk_size=500,overlap=50):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunks.append(text[start:end])
        start+=chunk_size-overlap
    return chunks
def load_all_pdfs(folder):
    all_chunks = []
    all_sources = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder, filename))
            chunks = chunk_text(text, chunk_size=800, overlap=100)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))
    return all_chunks, all_sources
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

import faiss
import numpy as np
def build_faiss_index(embeddings):
    dimensions=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimensions)
    index.add(np.array(embeddings))
    return index

def search(query, model, index, chunks, sources, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    result_sources = [sources[i] for i in indices[0]]
    return results, result_sources
def generate_answer(query,context_chunks,model_name="llama3.2"):
    context="\n\n".join(context_chunks)
    prompt=f"""Answer the question using only the context below.
    If the answer is not the context, say "I couldn't find that in the document."
    Context:
    {context}
    Question: {query}
    Answer"""
    response=ollama.chat(
        model=model_name,
        messages=[{"role":"user","content":prompt}]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    chunks, sources = load_all_pdfs("data")
    print(f"Number of chunks: {len(chunks)}")

    embeddings = get_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)

    query = "What frontend technology was used?"
    results, result_sources = search(query, model, index, chunks, sources, top_k=5)
    print("\n--- Retrieved Chunks ---")
    for i, r in enumerate(results):
        print(f"\nChunk {i+1} (from {result_sources[i]}):")
        print(r[:200], "...")

    answer = generate_answer(query, results)
    print("\n--- Answer ---")
    print(answer)