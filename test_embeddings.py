from pypdf import PdfReader
from sentence_transformers import SentenceTransformer,util

# load pdf nd extract text::
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
