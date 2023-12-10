import streamlit as st
import os
from emotion_recognition import predict_single_sample
from utils import save_file, transcribe_audio, emotion_emojis


# Streamlit page configuration
st.set_page_config(page_title="Real-Time Speech-to-Text", layout="wide")

# Page title
st.title("VoiceSense: Speech-to-Text & Emotion Analysis 🗣️❤️")

# Function to list audio samples
def list_audio_samples(directory):
    return [file for file in os.listdir(directory) if file.endswith(('.wav', '.mp3', '.ogg'))]

# User choice for audio source
audio_source = st.radio("Choose your audio source", ("Upload Audio", "Select from Samples"))

# Placeholder for displaying audio file and transcribed text
audio_display = st.empty()
transcription_display = st.empty()

# Handling the audio file upload
if audio_source == "Upload Audio":
    # File uploader for audio files
    audio_file = st.file_uploader("Upload your Audio File", type=['wav', 'mp3', 'ogg'])
    if audio_file is not None:
        # Save the audio file
        full_audio_path = save_file(audio_file)

        audio_file_size = f"{round(audio_file.size / 1024, 2)} KB"

        file_details = {'filename':audio_file.name, 'filetype':audio_file.type, 'filesize': audio_file_size}
        st.markdown("**Audio File Details:** 📄")
        st.write(file_details)
        
        audio_bytes = audio_file.read()
        # Display the audio file
        audio_display.audio(audio_bytes, format="audio/wav")
        # Transcribe the audio file (this would be real-time in the complete implementation)
        transcription = transcribe_audio(f"./audio_files/{audio_file.name}")
        
        # Display the transcription
        st.markdown("### **Transcribed Text:** 📝")
        st.write(transcription)

        # Display the sentiment analysis result
        st.markdown("### **Emotion Analysis:** 😊")
        # Display a warning about the model's limitations
        st.warning(
            "⚠️ Please note: The emotion recognition model is trained on a limited dataset "
            "and for a few epochs, which may affect its accuracy. It should be used as a "
            "guideline only and might not always reflect the true emotional context."
        )


        # Emotion analysis
        emotion_result = predict_single_sample(f"./audio_files/{audio_file.name}")
        if emotion_result is not None:
            pred, confidence = emotion_result
            # Get the corresponding emoji for the predicted emotion
            emoji = emotion_emojis.get(pred, '❓')  # Default to question mark if emotion not found
            st.write(f'Emotion: {pred} {emoji} (Confidence: {confidence:.2f})')
        else:
            st.write("Emotion analysis failed for this sample. 😕")

elif audio_source == "Select from Samples":
    # Display samples for selection
    audio_samples = list_audio_samples('audio_files')
    selected_sample = st.selectbox("Choose a sample to test:", audio_samples)

    if selected_sample:
        # Save the audio file
        audio_file_path = os.path.join('audio_files', selected_sample)
        audio_display.audio(audio_file_path, format=f"audio/{selected_sample.split('.')[-1]}")

        audio_file_size = f"{round(os.path.getsize(audio_file_path) / 1024, 2)} KB"
        file_details = {'filename':selected_sample, 'filetype':selected_sample.split('.')[-1], 'filesize': audio_file_size}

        st.markdown("**Audio File Details:** 📄")
        st.write(file_details)
        
        # Display the audio file
        audio_display.audio(os.path.join('audio_files', selected_sample), format=f"audio/{selected_sample.split('.')[-1]}")
        # Transcribe the audio file (this would be real-time in the complete implementation)
        transcription = transcribe_audio(os.path.join('audio_files', selected_sample))
        
        # Display the transcription
        st.markdown("### **Transcribed Text:** 📝")
        st.write(transcription)

        # Display the sentiment analysis result
        st.markdown("### **Emotion Analysis:** 😊")
        # Display a warning about the model's limitations
        st.warning(
            "⚠️ Please note: The emotion recognition model is trained on a limited dataset "
            "and for a few epochs, which may affect its accuracy. It should be used as a "
            "guideline only and might not always reflect the true emotional context."
        )


        # Emotion analysis
        emotion_result = predict_single_sample(os.path.join('audio_files', selected_sample))
        if emotion_result is not None:
            pred, confidence = emotion_result
            # Get the corresponding emoji for the predicted emotion
            emoji = emotion_emojis.get(pred, '❓')  # Default to question mark if emotion not found
            st.write(f'Emotion: {pred} {emoji} (Confidence: {confidence:.2f})')
        else:
            st.write("Emotion analysis failed for this sample. 😕")