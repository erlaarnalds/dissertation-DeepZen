import os
from mistralai import Mistral
import csv
from obj_models.client import Client
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class MistralAPIClient(Client):
    def __init__(self, model_name, finetuned=False):
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)

        rate_limit_per_min = 300 - 50
        self.delay = 60.0 / rate_limit_per_min

        super().__init__(model_name, finetuned)

        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def rank_utterance(self, system_msg, utterance):

        input_text = "[INST]" + system_msg + "[/INST]\n" + utterance

        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages = [{"role":'user', "content": input_text}],
        )

        return chat_response.choices[0].message.content.lower()
