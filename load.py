from pypdf import PdfReader

reader = PdfReader("data/Congestion_Control.pdf")
print(f"Number of pages: {len(reader.pages)}")
full_text = ""
for page in reader.pages:
    full_text+=page.extract_text()
print(full_text)
# first_page = reader.pages[0]
# print(first_page.extract_text())
chunk_size=500
chunks=[]
for i in range(0,len(full_text),chunk_size):
    chunk=full_text[i:i+chunk_size]
    chunks.append(chunk)
print(f"Number of chunks: {len(chunks)}")
print(chunks[0])
