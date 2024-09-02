import os
from pydub import AudioSegment

# Function to convert WAV to MP3
def convert_wav_to_mp3(wav_file_path, mp3_file_path):
    # Load the WAV file
    audio = AudioSegment.from_wav(wav_file_path)
    
    # Export the audio as an MP3 file
    audio.export(mp3_file_path, format="mp3")
    print(f"Converted {wav_file_path} to {mp3_file_path}")

def convert_all_wav_to_mp3(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            wav_file_path = os.path.join(directory, filename)
            mp3_dir = os.path.join(f"{directory}/mp3/", filename)
            mp3_file_path = os.path.splitext(mp3_dir)[0] + ".mp3"
            convert_wav_to_mp3(wav_file_path, mp3_file_path)

# Example usage
directory_path = "voice_samples/audio"  # Replace with the directory containing your WAV files
convert_all_wav_to_mp3(directory_path)