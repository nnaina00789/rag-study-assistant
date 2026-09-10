
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

def search(query,model,index,chunks,top_k=3):
    query_embedding=model.encode([query])
    distances,indices=index.search(np.array(query_embedding),top_k)
    results=[chunks[i] for i in indices[0]]
    return results


if __name__=="__main__":
    text=load_pdf("NAINA_12415429_summer.pdf")
    chunks=chunk_text(text)
    print(f"Number of chunks: {len(chunks)}")

    embeddings = get_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    index=build_faiss_index(embeddings)

    query="what is this project about?"

    results=search(query,model,index,chunks,top_k=3)
    for i,r in enumerate(results):
        print(f"\n-----Result{i+1}---")
        print(r)