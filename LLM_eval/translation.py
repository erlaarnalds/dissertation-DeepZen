from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pandas as pd
import csv
from datasets import Dataset
import ast


# def translate_text(text, model, tokenizer, device):

#     translate = pipeline("translation_XX_to_YY",model=model,tokenizer=tokenizer,device=device,src_lang="en_XX",tgt_lang="is_IS")
#     target_seq = translate(text, src_lang="en_XX",tgt_lang="is_IS",max_length=1024)
#     return target_seq[0]['translation_text'].strip('YY ')




# device = torch.cuda.current_device() if torch.cuda.is_available() else -1
# print(torch.cuda.is_available())

# tokenizer = AutoTokenizer.from_pretrained("mideind/nmt-doc-en-is-2022-10",src_lang="en_XX",tgt_lang="is_IS")

# model = AutoModelForSeq2SeqLM.from_pretrained("mideind/nmt-doc-en-is-2022-10")


# task = "ABSA"
# dataset = "uabsa_rest16"

# # Load the dataset with pandas
# df = pd.read_csv(f"datasets_local/{task}/english/{dataset}/test.csv")


# with open(f"datasets_local/{task}/icelandic/{dataset}/test2.csv", "w+", newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["original_id", "utterance", "label"])

#     for index, row in df.iterrows():
#         comment = row['text']
#         translated_comment = translate_text(comment, model, tokenizer, device)

#         label_arr = ast.literal_eval(row['label_text'])
#         translated_labels = []
#         for aspect, label in label_arr:
#             translated_aspect = translate_text(aspect, model, tokenizer, device)
#             translated_labels.append((translated_aspect, label))

#         writer.writerow([row['original_index'], translated_comment, repr(translated_labels)])


import os
from google.cloud import translate_v2 as translate
import pandas as pd
import csv

# Set environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.google_service_key.json'

# Initialize Google Translate client
translate_client = translate.Client()

# Function to translate text using Google Translate
def translate_text(text, target_language='is'):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# Load the dataset with pandas
df = pd.read_csv("data/MAST/english/stance/test.csv")

# Open output CSV file
with open("data/MAST/icelandic/stance/test.csv", "w+", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    #original_index,domain,text,label,label_text
    writer.writerow(["original_id", "domain", "utterance", "label"])

    # Iterate over the dataset and translate each row
    for index, row in df.iterrows():
        comment = row['text']
        translated_comment = translate_text(comment, target_language='is')

        domain = row['domain']
        translated_domain = translate_text(domain, target_language='is')

        writer.writerow([row['original_index'], translated_domain, translated_comment, row['label_text']])

print("Translation completed and saved to CSV.")