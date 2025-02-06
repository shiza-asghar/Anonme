import streamlit as st
import pyttsx3
from io import BytesIO
import base64

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index for different voices if needed
engine.setProperty('rate', 150)  # Speed of speech

# Define the AI response function
def ai_response(user_input):
    # Replace with your ML model's API or logic
    return f"Hello! I'm Elon Musk. I know John : '{user_input}' "

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

# Streamlit UI
st.set_page_config(page_title="AI Avatar - Elon Musk", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .center-image {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }

        .title h1 {
            color: #FFFFFF;
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .avatar {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
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
        .input-container {
            display: flex;
            gap: 10px;
        }
        .voice-button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
        }
        .voice-button:hover {
            background-color: #388E3C;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title"><h1>AI Avatar - Elon Musk</h1></div>', unsafe_allow_html=True)

# Display local avatar image
# Display local avatar image

# Center the image using st.markdown with custom CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="elon.jpg" width="300" alt="Elon Musk Avatar">
    </div>
    """,
    unsafe_allow_html=True,
)
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
        # Get the AI response
        response = ai_response(user_input)

        # Display the response
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

# Footer note
st.write("---")
st.markdown(
    '<p style="text-align:center; color:gray; font-size:14px;">This is a demo application using Streamlit.</p>',
    unsafe_allow_html=True,
)
