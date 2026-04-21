"""
Load and prepare the Spotify dataset for RAG
"""

import pandas as pd

print("=" * 50)
print("📀 LOADING SPOTIFY DATASET")
print("=" * 50)

# Load the dataset
df = pd.read_csv("spotify_songs.csv")
print(f"✓ Loaded {len(df)} songs!")

# See what columns we have
print(f"\n📊 Columns available:")
for col in df.columns:
    print(f"   - {col}")

# Preview first few songs
print("\n🎵 Sample songs:")
print(df[['song', 'artist', 'Genre', 'emotion']].head(10))

# Create RAG-ready text chunks
print("\n📝 Creating RAG chunks...")

rag_chunks = []

for _, song in df.head(1000).iterrows():  # Start with 1000 songs
    # Create a rich description for each song
    chunk = f"""
TITLE: {song['song']}
ARTIST: {song['artist']}
GENRE: {song['Genre']}
YEAR: {song['Release Date']}
EMOTION: {song['emotion']}
ENERGY: {song['Energy']}/100
DANCEABILITY: {song['Danceability']}/100
POSITIVENESS: {song['Positiveness']}/100
POPULARITY: {song['Popularity']}/100

DESCRIPTION: This {song['Genre']} song by {song['artist']} has 
{song['Energy']}% energy and is {song['Danceability']}% danceable.
The emotional tone is {song['emotion']}.

---END OF SONG---
"""
    rag_chunks.append(chunk)

# Save chunks to file
with open("rag_song_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rag_chunks))

print(f"✓ Created {len(rag_chunks)} RAG chunks")
print("\n✅ Ready to build vector database!")

# Show some statistics
print("\n📊 EMOTION DISTRIBUTION:")
emotion_counts = df['emotion'].value_counts()
for emotion, count in emotion_counts.head(5).items():
    print(f"   {emotion}: {count} songs")