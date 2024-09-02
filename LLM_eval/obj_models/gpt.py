from openai import OpenAI
import os
import csv
from obj_models.client import Client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class GPTClient(Client):
    def __init__(self, model_name, finetuned=False):
        self.client = OpenAI()
        if model_name[:2] == "ft":
            finetuned = True

        super().__init__(model_name, finetuned)

        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def rank_utterance(self, system_msg, utterance):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": utterance}
            ]
        )

        return response.choices[0].message.content.lower()