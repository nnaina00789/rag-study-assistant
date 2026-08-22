from pypdf import PdfReader
from sentence_transformers import SentenceTransformer,util

# load pdf and extract text::

reader=PdfReader("data/Congestion_Control.pdf")
full_text=""
for page in reader.pages:
    full_text+=page.extract_text()

# chunk the text

chunk_size=500
chunks=[]
for i in range(0,len(full_text),chunk_size):
    chunk=full_text[i:i+chunk_size]
    chunks.append(chunk)
print(f"Number of chunks: {len(chunks)}")

# embeddings of real chunks instead of dummy sentences

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks)
print(f"Shape of embeddings: {embeddings.shape}")

question="What is congestion control?"
question_embedding=model.encode([question])
similarities=util.cos_sim(question_embedding,embeddings)
print(f"Similarities: {similarities}")

best_match_index=similarities.argmax()
print(f"Best match index: {best_match_index}")
print(f"Best match chunk: {chunks[best_match_index]}")