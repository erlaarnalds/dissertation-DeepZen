from transformers import VitsModel, VitsTokenizer
import torch
import scipy
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, Audio
import os

voice = "alfur"
hf_user = "erlaka"

dataset_dir = f"data/talromur/{voice}"
create = True

if create:
    #create HF model
    print("creating HF model...")

    df = pd.read_csv(f'{dataset_dir}/index.tsv', sep='\t', header=None, names=['audio_id', 'text', 'text_id', 'standardized_text'])

    # Verify the data structure
    print(df.head())

    # Create the audio path column
    audio_dir = f"{dataset_dir}/audio"
    df['audio'] = df['audio_id'].apply(lambda x: f"{audio_dir}/{x}.wav")

    # Create a dictionary with the columns you want
    data_dict = {
        'audio': df['audio'].tolist(),
        'text': df['text'].tolist(),
        'text_id': df['text_id'].tolist(),
        'standardized_text': df['standardized_text'].tolist()
    }

    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("audio", Audio())


    dataset.save_to_disk(f"{dataset_dir}/hf_db")
    dataset.push_to_hub(f"{hf_user}/talromur_{voice}", private=True)

else:
    print(f"using saved dataset: {dataset_dir}/hf_db")
    dataset = load_dataset(f"{hf_user}/talromur_{voice}")