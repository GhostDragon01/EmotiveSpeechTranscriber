import os
import whisper

# Initialize Whisper model
model = whisper.load_model("base")

emotion_emojis = {
    'Colere': '😠',    # Anger
    'Degout': '🤢',    # Disgust
    'Joie': '😄',      # Joy
    'Neutre': '😐',    # Neutral
    'Peur': '😨',      # Fear
    'Surprise': '😲',  # Surprise
    'Tristesse': '😢'  # Sadness
}


# Function to save audio file
def save_file(sound_file):
    # Directory where audio files will be saved
    audio_dir = 'audio_files'
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Full path for the audio file
    full_path = os.path.join(audio_dir, sound_file.name)

    # Save the audio file
    with open(full_path, 'wb') as f:
        f.write(sound_file.getbuffer())

    return full_path  # Return the full path of the saved file

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Check if the file exists
    if not os.path.exists(audio_file):
        print("File not found:", audio_file)
        return "File not found"
    
    # Process and transcribe the audio file using Whisper
    result = model.transcribe(audio_file)
    return result["text"]
