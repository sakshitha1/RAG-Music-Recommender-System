# I noticed some songs have mismatched mood labels
# Example: "Happy Happy Birthday Song" is marked as sadness
# This is because the dataset used AI to detect mood from lyrics

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import warnings
warnings.filterwarnings('ignore')

print("Loading models...")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("songs_collection")

# LLM from Hugging Face
print("Loading LLM (this takes 1-2 minutes first time)...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print("Models loaded.")

def get_song_info(text):
    """Extract basic info from song text"""
    title_match = re.search(r"TITLE: (.*?)\n", text)
    artist_match = re.search(r"ARTIST: (.*?)\n", text)
    genre_match = re.search(r"GENRE: (.*?)\n", text)
    emotion_match = re.search(r"EMOTION: (.*?)\n", text)
    
    return {
        "title": title_match.group(1) if title_match else "Unknown",
        "artist": artist_match.group(1) if artist_match else "Unknown",
        "genre": genre_match.group(1) if genre_match else "Unknown",
        "emotion": emotion_match.group(1) if emotion_match else "Unknown"
    }

def search_songs(question, k=3):
    """Search for similar songs in database"""
    q_emb = embed_model.encode(question, normalize_embeddings=True)
    
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k
    )
    
    return results['documents'][0]

def generate_answer(question, context_songs):
    """Generate answer using LLM - FIXED VERSION"""
    
    # Build context string
    context_text = ""
    for i, song in enumerate(context_songs, 1):
        info = get_song_info(song)
        context_text += f"{i}. {info['title']} by {info['artist']} - {info['genre']}, {info['emotion']} mood\n"
    
    # Improved prompt
    prompt = f"""Based on these songs, answer the question.

Question: {question}

Songs:
{context_text}

Write a short recommendation:"""

    # Generate WITHOUT temperature parameter (to avoid warning)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm.generate(
        inputs.input_ids, 
        max_length=200,
        do_sample=False  # Changed from True to avoid temperature warning
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # If answer is empty or too short, provide fallback
    if len(answer) < 10:
        answer = f"Based on your request, here are some songs:\n"
        for i, song in enumerate(context_songs, 1):
            info = get_song_info(song)
            answer += f"{i}. {info['title']} by {info['artist']} ({info['emotion']} mood)\n"
    
    return answer

def ask(question, k=3):
    """Main function - query the RAG system"""
    print(f"\nQuestion: {question}")
    print("Searching database...")
    
    songs = search_songs(question, k)
    
    print(f"Found {len(songs)} relevant songs")
    
    answer = generate_answer(question, songs)
    
    print("\n" + "="*50)
    print("ANSWER:")
    print(answer)
    print("\n" + "="*50)
    print("RETRIEVED SONGS:")
    for i, song in enumerate(songs, 1):
        info = get_song_info(song)
        print(f"{i}. {info['title']} - {info['artist']} ({info['emotion']})")
    
    return answer, songs

if __name__ == "__main__":
    test_q = "Give me happy upbeat songs"
    ask(test_q)