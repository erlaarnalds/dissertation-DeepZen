
from openai import OpenAI, APITimeoutError
import os
import time
import csv

import os
import csv
from obj_models.client import Client


class LlamaCppClient(Client):
    def __init__(self, model_name, finetuned=False):
        if "llama" in model_name:
            self.port = 8080
        elif "gemma" in model_name:
            self.port = 8081

        base_url=f"http://localhost:{self.port}/v1"
        api_key="sk-no-key-required"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        super().__init__(model_name, finetuned)

        

    def rank_utterance(self, system_msg, utterance, timeout=100):

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": utterance}
            ],
            timeout=timeout
        )

        return response.choices[0].message.content.lower().strip()
