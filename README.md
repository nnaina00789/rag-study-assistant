# RAG Study Assistant

A self-study helper that answers questions based on my own class notes and PDFs, instead of generic internet answers.

## How it works
I upload my study material (notes, PDFs). The system breaks it into chunks, converts them into embeddings (numeric representations of meaning), and stores them in a vector database. When I ask a question, it finds the most relevant chunks from my notes and uses an LLM to generate an answer grounded in that content.

## Tech Stack
- Python
- LangChain / LlamaIndex
- FAISS / ChromaDB (vector database)
- Sentence-Transformers (embeddings)
- FastAPI (API layer)

## Status
Day 1: Project setup only. Architecture planned, folder + git repo initialized. Build starts next.