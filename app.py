%%writefile app.py
import streamlit as st
import google.generativeai as genai
import os
from io import BytesIO
import tempfile
# Choose either pypdf or PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
import librosa
import numpy as np
import whisper


#store the gemini api key
GOOGLE_API_KEY = "Insert your key here"
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model you want to use
model = genai.GenerativeModel('gemini-2.0-flash')

# Load Whisper model
whisper_model = whisper.load_model("base")

# Define Pre-defined Prompts (now more general)
BASE_PROMPTS = {
    "Short Notes": "Generate short notes based on the content. Provide a crisp summary of the content as well.",
    "Detailed Notes": "Generate detailed notes based on the content. List the sub headings clearly and in bulleted form.",
    "Question Answers": "Generate 5 short answer based questions from the content. Give a question followed by its answer in upto 100 words.",
    "Colour Coded ": "Generate color coded notes for the content, put headings in bold and contrasting color for the content. Use emojis to color code."
}

def process_pdf(uploaded_file, base_prompt):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if text:
            full_prompt = f"{base_prompt}\n\nContent:\n{text}"
            response = model.generate_content([full_prompt])
            return response.text
        else:
            return "Error: Could not extract text from the PDF."
    except Exception as e:
        return f"Error processing PDF: {e}"

def process_audio(uploaded_file, base_prompt):
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        ## components of audio for technical analysis
        ## Load the audio file using librosa for audio features
        # y, sr = librosa.load(tmp_path)
        ## Get audio duration
        # duration = librosa.get_duration(y=y, sr=sr)
        # # Extract some basic audio features
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Transcribe audio using Whisper
        result = whisper_model.transcribe(tmp_path)
        transcribed_text = result["text"]

        # Clean up the temporary file
        os.unlink(tmp_path)

        # Generate response using Gemini
        full_prompt = f"{base_prompt}\n\nContent:\n{transcribed_text}"
        response = model.generate_content([full_prompt])
        return response.text

    except Exception as e:
        return f"Error processing audio: {e}"

## streamlit app styling

st.set_page_config(layout="centered")
st.markdown("""
<style>
  body{
    font-size: 16px;
    text-align: left;
  }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
st.markdown("<h1 style='text-align: center;'>Note-fy</h1>", unsafe_allow_html=True)
st.markdown("***")

# file uploader
uploaded_file = st.file_uploader("", type=["pdf", "mp3"])
st.markdown("<p style='text-align: center;'>(Select options from sidebar to personalize your notes)</p>", unsafe_allow_html=True)
st.markdown("***")

selected_prompt = st.sidebar.radio("Choose personalization:", list(BASE_PROMPTS.keys()))

if uploaded_file is not None:
    st.subheader("Uploaded File:")
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    #centering the submit button
    col1, col2, col3 , col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        center_button = st.button('Submit')

    if center_button:
        st.markdown("***")
        with st.spinner("Processing... This may take a few moments."):
            base_prompt = BASE_PROMPTS[selected_prompt]
            output = ""

            if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
                output = process_pdf(uploaded_file, base_prompt)
            elif uploaded_file.type == "audio/mpeg" or uploaded_file.name.endswith(".mp3"):
                output = process_audio(uploaded_file, base_prompt)
            else:
                output = "Unsupported file type. Please upload a .pdf or .mp3 file."


            st.subheader("Output:")
            container = st.container(border=True)
            container.write(output)
            st.slider("Satsfaction level", 0, 5, disabled=False)

else:
  st.markdown("<p style='text-align: center;'>Please select a file!!</p>", unsafe_allow_html=True)
