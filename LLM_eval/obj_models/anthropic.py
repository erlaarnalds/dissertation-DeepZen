from anthropic import Anthropic
import os
import time
import csv
from obj_models.client import Client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class AnthropicClient(Client):
    def __init__(self, model_name, finetuned=False):
        self.client = Anthropic()

        super().__init__(model_name, finetuned)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def rank_utterance(self, system_msg, utterance):

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=128,
            system=system_msg,
            messages=[
            {"role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": utterance
                }
            ]}
            ]
        )
        
        return response.content[0].text.lower()
