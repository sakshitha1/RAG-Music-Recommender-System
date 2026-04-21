'''
Build ChromaDB vector database from song chunks
'''

import chromadb
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

print("=" * 50)
print("🏗️ BUILDING VECTOR DATABASE")
print("=" * 50)

# Load the RAG chunks
print("\n[1/4] Loading RAG chunks...")
with open("rag_song_chunks.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = content.split("---END OF SONG---")
chunks = [c.strip() for c in chunks if c.strip()]
print(f"✓ Loaded {len(chunks)} song chunks")

# Load embedding model
print("\n[2/4] Loading embedding model...")
print("   (This downloads a small AI model - about 80MB)")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Model loaded successfully")

# Create embeddings
print("\n[3/4] Creating embeddings...")
embeddings = []
for chunk in tqdm(chunks, desc="   Progress"):
    emb = embedding_model.encode(chunk, normalize_embeddings=True)
    embeddings.append(emb)

print(f"✓ Created {len(embeddings)} embeddings")

# Store in ChromaDB
print("\n[4/4] Storing in ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("songs_collection")
    print("   Removed existing collection")
except:
    pass

collection = client.create_collection("songs_collection")

for i, (chunk, emb) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks), desc="   Adding to DB")):
    collection.add(
        documents=[chunk],
        embeddings=[emb.tolist()],
        ids=[str(i)]
    )

print(f"\n✅ Database built successfully!")
print(f"   Total songs in database: {collection.count()}")