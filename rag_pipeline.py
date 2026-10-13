import os
import time
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


from dotenv import load_dotenv

load_dotenv()

def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {query}

Answer:"""

def generate_answer_ollama(query, context_chunks, model_name="llama3.2"):
    import ollama
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": build_prompt(query, context_chunks)}]
    )
    return response["message"]["content"]

GEMINI_MODELS = ["gemini-3.6-flash","gemini-3.5-flash","gemini-3.1-flash-lite"]

def generate_answer_gemini(query, context_chunks):
    from google import genai
    from google.genai import errors
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = build_prompt(query, context_chunks)
    last_error = None
    for round_number in range(3):
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except errors.APIError as e:
                last_error = e
                if e.code < 500 and e.code != 404:
                    raise
        if round_number < 2:
            time.sleep(3 * (round_number + 1))
    raise last_error

def generate_answer(query, context_chunks):
    if os.getenv("LLM_BACKEND", "ollama") == "gemini":
        return generate_answer_gemini(query, context_chunks)
    return generate_answer_ollama(query, context_chunks)

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