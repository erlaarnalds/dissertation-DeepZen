import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import csv
from obj_models.client import Client


class MistralAPIClient(Client):
    def __init__(self, model_name, finetuned=False):
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = MistralClient(api_key=api_key)

        super().__init__(model_name, finetuned)

        

    def rank_utterance(system_msg, utterance):

        input_text = "[INST]" + system_msg + "[/INST]\n" + utterance

        chat_response = self.client.chat(
            model=self.model_name,
            messages=[ChatMessage(role="user", content=input_text)]
        )

        return chat_response.choices[0].message.content.lower()
