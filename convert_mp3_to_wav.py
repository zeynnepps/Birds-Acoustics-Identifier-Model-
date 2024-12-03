import os
from pydub import AudioSegment

def convert_mp3_to_wav(file_path):
    """Convert a single MP3 file to WAV and remove the original MP3 file."""
    if file_path.endswith(".mp3"):
        wav_path = os.path.splitext(file_path)[0] + ".wav"  # Change extension to .wav
        try:
            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(file_path)
            audio.export(wav_path, format="wav")
            print(f"Converted: {file_path} -> {wav_path}")

            # Remove the original MP3 file
            os.remove(file_path)
            print(f"Deleted MP3 file: {file_path}")
            return wav_path
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    return file_path  # If not MP3, return original file_path

def convert_mp3_in_directory(data_dir):
    """Look for MP3 files and convert them to WAV."""
    mp3_files_found = False
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files_found = True
                file_path = os.path.join(root, file)
                convert_mp3_to_wav(file_path)

    if not mp3_files_found:
        print("No MP3 files found to convert.")

# Prompt user for the path to the data directory
data_dir = input("Please enter the path to the 'bird_sounds' directory: ")
convert_mp3_in_directory(data_dir)
