import gradio as gr
from rag_engine import ask, get_song_info

def process_query(question, k):
    if not question:
        return "Please enter a question.", ""
    
    try:
        answer, songs = ask(question, k)
        
        songs_text = ""
        for i, song in enumerate(songs, 1):
            info = get_song_info(song)
            songs_text += f"{i}. **{info['title']}** - {info['artist']}\n"
            songs_text += f"   Genre: {info['genre']} | Mood: {info['emotion']}\n\n"
        
        return answer, songs_text
        
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks(title="Music Recommender") as demo:
    gr.Markdown("# Music Recommendation System")
    gr.Markdown("Ask for song recommendations based on mood, genre, or energy level.")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="What kind of music do you want?",
                placeholder="Example: Give me sad songs for a rainy day",
                lines=3
            )
            k_value = gr.Slider(
                minimum=1, 
                maximum=5, 
                value=3, 
                step=1,
                label="Number of songs to retrieve"
            )
            submit = gr.Button("Get Recommendations", variant="primary")
        
        with gr.Column():
            answer = gr.Textbox(label="AI Response", lines=8, interactive=False)
            songs_display = gr.Textbox(label="Retrieved Songs", lines=10, interactive=False)
    
    submit.click(
        fn=process_query,
        inputs=[question, k_value],
        outputs=[answer, songs_display]
    )
    
    gr.Markdown("""
    ---
    ### Example questions:
    - "Give me happy upbeat songs"
    - "Sad emotional music for studying"
    - "High energy workout songs"
    """)

if __name__ == "__main__":
    demo.launch(share=True)