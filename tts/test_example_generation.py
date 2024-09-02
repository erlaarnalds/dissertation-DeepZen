from openai import OpenAI
import os
import pandas as pd
import csv
import shutil

from tts_openai import WhisperClient
from tts_tiro import TiroClient
from tts_mms import MMSClient


def load_env():
    """Helper function to read environment variables - where API keys are stored"""

    with open(".env", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value




def get_client(model_name):
    if model_name == "whisper":
        return WhisperClient("onyx")
    
    if model_name == "tiro":
        return TiroClient("Alfur_v2")
    
    if model_name == "mms":
        return MMSClient("facebook/mms-tts-isl", "mms", 2)
    
    if model_name == "mms_finetuned":
        return MMSClient("erlaka/mms-tts-isl-finetuned-dilja", "mms_finetuned", 4)



def generate_samples(client, db_samples, output_dir):

    #question to pose to reviewers:
    question = "Gefðu þessari rödd einkunn út frá því hve vel þér líkar við hana."

    for ind, row in db_samples.iterrows():
        
        text = row["text"]
        file_name = f"{client.model_name}_{row['audio_id']}.wav"
        
        location = f"{output_dir}/audio/{file_name}"

        client.generate(text, location)

        with open(f'{output_dir}/index.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
            # file_name, S/R (synthetic/recorded), read text, voice_id, sentence_id, model_id, suffix, question
            writer.writerow([file_name, "S", text, client.model_id, ind, client.model_id, "", question])


def generate_baseline(phrases, baseline_dir, output_dir):

    #question to pose to reviewers:
    question = "Gefðu þessari rödd einkunn út frá því hve vel þér líkar við hana."

    baseline_data = pd.read_csv(f'{baseline_dir}/index.tsv', sep='\t', header=None, names=['audio_id', 'text', 'text_id', 'standardized_text'])
    sampled_rows = baseline_data.loc[baseline_data['text_id'].isin(phrases)]
    
    for ind, row in sampled_rows.iterrows():
        text = row['text']

        file_name = f"{row['audio_id']}.wav"

        with open(f'{output_dir}/index.csv', 'a', newline='') as csvfile:
            shutil.copyfile(f"{baseline_dir}/audio/{file_name}", f"{output_dir}/audio/{file_name}")
            writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
            # file_name, S/R (synthetic/recorded), read text, voice_id, sentence_id, model_id, suffix, question
            writer.writerow([file_name, "R", text, 0, ind, 0, "", question])



if __name__ == "__main__":
    load_env()

    dataset_dir = "data/talromur/dilja"
    baseline_dir = "data/talromur/bui"
    index_data = pd.read_csv(f'{dataset_dir}/index.tsv', sep='\t', header=None, names=['audio_id', 'text', 'text_id', 'standardized_text'])

    # phrases selected as being lyrical/narrative as well as existing in both datasets
    phrases = ["t002419", "t020691", "t008874", "t005110", "t003349", "t00887", "t019713", "t008181", "t019845", "t000703", "t002693"]
    sampled_rows = index_data.loc[index_data['text_id'].isin(phrases)]
    

    models = ["baseline", "whisper", "tiro", "mms", "mms_finetuned"]
    output_dir = "voice_samples"

    for model in models:
        print(model)
        client = get_client(model)

        if model == "baseline":
            generate_baseline(phrases, baseline_dir, output_dir)
        else:
            generate_samples(client, sampled_rows, output_dir)

        

