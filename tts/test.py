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
    i = 0
    for base_filename in os.listdir(directory):
        if base_filename.startswith("s"):
            if i == 10:
                break
            i += 1
            synth_file_name = f"mms_finetuned_{base_filename}"

            for filename in [base_filename, synth_file_name]:
                wav_file_path = os.path.join(directory, filename)
                mp3_dir = os.path.join(f"voice_samples/audio/dilja/", filename)
                mp3_file_path = os.path.splitext(mp3_dir)[0] + ".mp3"
                convert_wav_to_mp3(wav_file_path, mp3_file_path)

# Example
directory_path = "secs_evaluation_samples/dilja"  # Replace with the directory containing your WAV files
convert_all_wav_to_mp3(directory_path)