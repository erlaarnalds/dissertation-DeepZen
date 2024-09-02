
from openai import OpenAI, APITimeoutError
import os
import time
import csv
import tiktoken

import os
import csv
from obj_models.client import Client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class LlamaCppClient(Client):
    def __init__(self, model_name, finetuned=False):
        if "llama" in model_name:
            self.port = 8080
        elif "gemma" in model_name:
            self.port = 8081

        base_url=f"http://localhost:{self.port}/v1"
        api_key="sk-no-key-required"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        super().__init__(model_name, finetuned)

        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def rank_utterance(self, system_msg, utterance):

        # tokens = self.encoding.encode(system_msg)
        # print(f"Token count: {len(tokens)}")
        # utterance_tokens = self.encoding.encode(utterance)
        # print(f"Token count: {len(utterance_tokens)}")

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": utterance}
            ],
            max_tokens=128
        )

        return response.choices[0].message.content.lower().strip()
