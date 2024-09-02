from pathlib import Path
from openai import OpenAI

class WhisperClient():
    def __init__(self, voice):
        self.client = OpenAI()
        self.voice = voice
        self.model_name = "whisper"
        self.model_id = 3

    def generate(self, text, output_dir):

        speech_file_path = Path(__file__).parent / output_dir

        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text,
            response_format="wav"
        )

        response.stream_to_file(speech_file_path)
        
        
