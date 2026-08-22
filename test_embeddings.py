from sentence_transformers import SentenceTransformer,util
model=SentenceTransformer("all-MiniLM-L6-v2")
sentence1="The dog ran fast"
sentence2="The cnaine sprinted quickly"
sentence3="I love cooking pasta"
embeddings=model.encode([sentence1,sentence2,sentence3])
print(f"Shape of embeddings: {embeddings.shape}")
print(embeddings [0][:5])
similarity_1_2=util.cos_sim(embeddings[0],embeddings[1])
similarity_1_3=util.cos_sim(embeddings[0],embeddings[2])
print(f"similarity between dog/canine: {similarity_1_2}")
print(f"similarity between dog/pasta: {similarity_1_3}")
