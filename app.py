# Import necessary libraries
import os
import tempfile
import whisper
from gtts import gTTS
from groq import Groq
import gradio as gr

# Set up the Groq API client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Environment variable set in Gradio

# Load the Whisper model
model = whisper.load_model("base")

# Function to handle the real-time voice-to-voice conversation
def career_guidance_bot(audio):
    # Transcribe audio to text using Whisper with English forced
    result = model.transcribe(audio, language="en")
    user_input = result["text"]
    
    # Send transcribed text to Groq's LLM with an English prompt constraint
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"{user_input}. Respond in English, please."}],
        model="llama-3.2-11b-vision-preview"
    )
    output_text = response.choices[0].message.content

    # Convert response text to speech using gTTS
    tts = gTTS(output_text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        audio_path = temp_audio_file.name

    # Return the bot response text and audio path
    return output_text, audio_path

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align: center;'>Your Career Success Partner</h1>")
    gr.Markdown("This app provides real-time career guidance and interview practice support. Just speak your question to get audio responses.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Click and Speak") 
        output_text = gr.Textbox(label="Transcribed Text")
        output_audio = gr.Audio(label="Bot's Response")

    # Interface for audio input, transcribed text, and audio output
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=career_guidance_bot, inputs=audio_input, outputs=[output_text, output_audio])

    # Signature at the bottom of the app
    gr.Markdown("<p style='text-align: right; color: grey;'>This App is made by Syed Amjad Ali</p>")

# Launch the Gradio app
app.launch()
