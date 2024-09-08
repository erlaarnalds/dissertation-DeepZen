import torch
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import shutil
import numpy as np
from tts_mms import MMSClient
from tts_tiro import TiroClient
import torchaudio


def load_env():
    """Helper function to read environment variables - where API keys are stored"""

    with open(".env", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value



def extract_embedding(encoder, wav_path):
    # Load the wav file and extract the speaker embedding
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)
    return embedding



def generate_samples(client, row, output_dir):

    text = row["text"]
    file_name = f"{client.model_name}_{row['audio_id']}.wav"
    location = f"{output_dir}/{file_name}"

    if not os.path.exists(location):
        client.generate(text, location)



# def get_embeddings(client, samples, location, ismodel=False):

#     embeddings = []

#     for ind, row in samples.iterrows():

#         if ismodel:
#             file_name = f"{location}/{client.model_name}_{row['audio_id']}.wav"
#         else:
#             file_name = f"{location}/{row['audio_id']}.wav"
        
#         file_embeddings = extract_embedding(encoder, file_name)
#         embeddings.append(file_embeddings)
    
#     return np.asarray(embeddings)


def evaluate(speaker_encoder, client, dataset_dir, output_dir, generate = False):

    index_data = pd.read_csv(f'{dataset_dir}/index.tsv', sep='\t', header=None, names=['audio_id', 'text', 'text_id', 'standardized_text'])

    N = 100

    sampled_rows = index_data.sample(n=N, random_state=42)
    
    if generate:
        i = 0
        for ind, row in sampled_rows.iterrows():
            print(i)
            i += 1

            
            generate_samples(client, row, output_dir)


            file_name = f"{row['audio_id']}.wav"
            file_path = f"{output_dir}/{file_name}"

            if not os.path.exists(file_path):
                shutil.copyfile(f"{dataset_dir}/audio/{file_name}", file_path)
    
    human_embeddings = []
    tts_embeddings = []

    for _, row in sampled_rows.iterrows():
        human_wav_path = f"{output_dir}/{row['audio_id']}.wav"
        tts_wav_path = f"{output_dir}/{client.model_name}_{row['audio_id']}.wav"
        
        human_embeddings.append(extract_embedding(speaker_encoder, human_wav_path))
        tts_embeddings.append(extract_embedding(speaker_encoder, tts_wav_path))
        

    human_embeddings = np.array(human_embeddings)
    tts_embeddings = np.array(tts_embeddings)
    print(human_embeddings.shape)
    print(human_embeddings.shape)
    

    # Calculate the similarity for each pair
    similarities = cosine_similarity(human_embeddings, tts_embeddings)

    show_top_three(similarities)

    print("Top 3:")
    top3 = np.argmax(similarities.diagonal())[:3]
    for ind in top3:
        print(f"{sampled_rows.iloc[ind]['audio_id']}: {similarities.diagonal()[ind]}")
    
    # Compute the average similarity
    average_similarity = np.mean(similarities.diagonal())
    std_dev = np.std(similarities.diagonal())
    return average_similarity, std_dev


if __name__ == "__main__":
    load_env()

    speaker_encoder = VoiceEncoder()

    # mms:
    dataset_dir = "data/talromur/dilja"
    output_dir = "secs_evaluation_samples/dilja"
    client = MMSClient("erlaka/mms-tts-isl-finetuned-dilja", "mms_finetuned", 4)
    average_similarity_mms, stddev_mms = evaluate(speaker_encoder, client, dataset_dir, output_dir, generate=True)
    
    print(f"Average SECS for MMS finetuned: {average_similarity_mms}")
    print(f"Standard deviation: {stddev_mms}")

    # Tiro:
    dataset_dir = "data/talromur/alfur"
    output_dir = "secs_evaluation_samples/alfur"
    client = TiroClient("Alfur_v2")
    average_similarity_tiro, stddev_tiro = evaluate(speaker_encoder, client, dataset_dir, output_dir, generate=True)

    print(f"Average SECS for Tiro: {average_similarity_tiro}")
    print(f"Standard deviation: {stddev_tiro}")





