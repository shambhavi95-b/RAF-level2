from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

embedding = model.encode("Hello world")

print("SUCCESS")
print("Embedding dimensions:", len(embedding))