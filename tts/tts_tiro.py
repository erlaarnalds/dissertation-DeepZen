import requests
import io
from pydub import AudioSegment

class TiroClient():
    def __init__(self, voice_id):
        self.base_url = "https://tts.tiro.is"
        self.voice_id = voice_id
        self.model_name = "tiro"
        self.model_id = 1

    
    def get_request(self, text):
        return {
            "Engine": "standard",
            "LanguageCode": "is-IS",
            "LexiconNames": [],
            "OutputFormat": "mp3",
            "SampleRate": "22050",
            "SpeechMarkTypes": [
                "word"
            ],
            "Text": text,
            "TextType": "text",
            "VoiceId": self.voice_id
        }

    def generate(self, text, output_dir):

        url = self.base_url + "/v0/speech"

        req = self.get_request(text)

        response =  requests.post(url, json=req)

        if response.status_code == 200:
            assert response.headers["Content-Type"] == "audio/mpeg", f"content type is {response.headers['Content-Type']}"
            
            mp3_data = io.BytesIO(response.content)
            audio = AudioSegment.from_file(mp3_data, format="mp3")

            audio.export(output_dir, format="wav")

        else:
            print("Request failed with status code ", response.status_code)
            print(response.content)
            return