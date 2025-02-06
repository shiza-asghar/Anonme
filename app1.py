import streamlit as st
import pyttsx3
from io import BytesIO
import requests
import base64

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index for different voices if needed
engine.setProperty('rate', 150)  # Speed of speech


import requests

def ai_response(user_input):
    """
    Sends the user input to the /chat endpoint of your API and returns the response.
    
    Args:
        user_input (str): The text input from the user.
        
    Returns:
        str: The response from the API.
    """
    url = "http://127.0.0.1:3356/chat"  # Your /chat endpoint URL
    try:
        # Send the user input to the API with 'query' as a key
        response = requests.post(url, json={"query": user_input})

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result.get("result", "No result received from API.")
        else:
            return f"Error {response.status_code}: {response.json().get('detail', 'Unknown error.')}"
    except Exception as e:
        return f"Failed to connect to the API. Error: {str(e)}"



# Function for text-to-speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to generate a voice note file
def generate_voice_note(text):
    audio_file = BytesIO()
    engine.save_to_file(text, audio_file)
    audio_file.seek(0)
    return audio_file

# Function to call the transcription API
def transcribe_audio(file):
    """
    Sends an audio file to the FastAPI `/transcribe` endpoint for transcription.
    Args:
        file (BytesIO): Audio file object to send.
    Returns:
        str: The transcribed text if successful, or an error message.
    """
    url = "http://127.0.0.1:3356/transcribe"
    try:
        # Send the file to the API
        response = requests.post(
            url,
            files={"file": ("audio_file", file, "audio/wav")}  # Don't set Content-Type manually
        )
        # Check the response status
        if response.status_code == 200:
            result = response.json()
            return result.get("result", "No transcription result received.")
        else:
            return f"Error {response.status_code}: {response.json().get('detail', 'Unknown error.')}"
    except Exception as e:
        return f"Failed to connect to the API. Error: {str(e)}"


# Streamlit UI
st.set_page_config(page_title="AI Avatar - Elon Musk", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .title h1 {
            color: #FFFFFF;
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .response-box {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            font-family: Arial, sans-serif;
            margin-top: 20px;
            color: #2e7d32;
        }
        .download-link {
            margin-top: 15px;
            font-size: 16px;
            text-decoration: none;
            color: #2e7d32;
        }
        .download-link:hover {
            color: #1b5e20;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title"><h1>AI Avatar - Elon Musk</h1></div>', unsafe_allow_html=True)

# Centered Avatar Image
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])  # Create five columns with a 1:1:2:1:1 ratio
with col3:  # Use the middle column
    st.image("elon.jpg", width=300, caption="AI Avatar: Elon Musk")

# Chat section with a text input and a voice note button
st.write("### Chat with Elon Musk")

# Create the input field and the button using columns
col1, col2 = st.columns([4, 1])  # 4:1 ratio for input and button
with col1:
    user_input = st.text_input("Enter your message:", placeholder="Type something...")
with col2:
    if st.button("🎤 Voice Note"):
        st.info("Voice note functionality will be implemented here (e.g., speech-to-text).")

# Button to process the input
if st.button("Send Message"):
    if user_input:
        # Get the AI response from your API
        response = ai_response(user_input)

        # Display the response from the API
        st.markdown(
            f'<div class="response-box">AI Avatar says: {response}</div>',
            unsafe_allow_html=True,
        )

        # Speak the response
        speak_text(response)

        # Generate and provide a downloadable voice note
        audio_file = generate_voice_note(response)
        b64_audio = base64.b64encode(audio_file.getvalue()).decode("utf-8")
        audio_html = f"""
        <a class="download-link" href="data:audio/wav;base64,{b64_audio}" download="response.wav">
            Download Voice Note
        </a>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.error("Please enter a message to chat.")


# Upload an audio file
st.write("### Upload an Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file (e.g., MP3, WAV, etc.)",
    type=["mp3", "wav", "m4a", "flac", "ogg", "aac"]
)

if uploaded_file:
    st.audio(uploaded_file, format=uploaded_file.type, start_time=0)

    # Transcribe button
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing..."):
            # Convert uploaded file to BytesIO for sending to FastAPI
            audio_bytes = BytesIO(uploaded_file.read())
            transcription = transcribe_audio(audio_bytes)
            st.success("Transcription Complete!")
            st.markdown(f"**Transcription Result:** {transcription}")

# Footer note
st.write("---")
st.markdown(
    '<p style="text-align:center; color:gray; font-size:14px;">This is a demo application using Streamlit.</p>',
    unsafe_allow_html=True,
)
