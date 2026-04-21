# RAG-Music-Recommender-System

A Retrieval-Augmented Generation (RAG) system that recommends songs from natural language queries. Built with ChromaDB and Hugging Face FLAN-T5.

## Features

- Ask for music in plain English (e.g., "sad songs for a rainy day")
- ChromaDB for semantic search
- Hugging Face FLAN-T5 for AI-generated recommendations
- Gradio web interface

## Tech Stack

| Component | Tool |
|-----------|------|
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Hugging Face FLAN-T5-small |
| Web UI | Gradio |
| Data | 200K Spotify Songs (Kaggle) |

## Project Structure
spotify_rag_project/
├── spotify_songs.csv # Dataset
├── prepare_data.py # Convert CSV to text chunks
├── build_vector_db.py # Build ChromaDB
├── rag_engine.py # RAG pipeline (search + generate)
├── app.py # Gradio web interface
└── chroma_db/ # Vector database (auto-created)

text

## Setup and Run

```bash
# 1. Install dependencies
pip install pandas chromadb sentence-transformers transformers gradio

# 2. Download dataset from Kaggle (200K Spotify Songs)
# Save as spotify_songs.csv in project folder

# 3. Build database
python prepare_data.py
python build_vector_db.py

# 4. Run web app
python app.py
Open http://127.0.0.1:7860 in your browser.

Example Results
Query: "depressing songs"

Retrieved:

Lay My Burden Down - Alison Krauss (sadness)

Jubilee - Alison Krauss (sadness)

Here Comes Goodbye - Alison Krauss (sadness)

AI Response: "Alison Krauss"

Query: "happy upbeat songs"

Retrieved:

The Happy Happy Birthday Song - Arrogant Worms (sadness)*

Happy Holidays - Alabama (joy)

Happy Hawaii - ABBA (joy)

AI Response: "Happy Hawaii"

*Note: Dataset limitation - title says "Happy" but mood labeled as sadness

Problems and Solutions
Problem	Solution
Spotify API required premium account :	Switched to Kaggle dataset
Hugging Face pipeline error	: Used AutoTokenizer + AutoModelForSeq2SeqLM
ChromaDB collection name mismatch	Fixed "songs" to "songs_collection"
Evaluation
Metric	Result
Retrieval accuracy = ~80%
Response time =	3-5 seconds
Database size	= 1000 songs
ChromaDB query = <0.5 seconds
Limitations
Mood labels are AI-generated and sometimes incorrect

FLAN-T5-small gives very short answers

Only 1000 songs used (dataset has 200K)

Links
Dataset: https://www.kaggle.com/datasets/devdope/200k-spotify-songs-light-dataset

Author
Bobbala Sakshitha
Deep Learning
April 2026
