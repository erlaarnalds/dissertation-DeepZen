from anthropic import Anthropic
import os
import time
import csv
from obj_models.client import Client


class AnthropicClient(Client):
    def __init__(self, model_name, finetuned=False):
        self.client = Anthropic()

        super().__init__(model_name, finetuned)

        
    def rank_utterance(self, system_msg, utterance):

        rate_limit_per_minute = 50
        delay = 60.0 / rate_limit_per_minute

        # slow down requests to prevent rate limiting
        time.sleep(delay)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
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
