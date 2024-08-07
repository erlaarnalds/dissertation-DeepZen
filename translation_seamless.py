from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import pandas as pd
import csv
from datasets import Dataset
import ast
import os
import csv

def translate_text(text_inp):
    text_inputs = processor(text = text_inp, src_lang="eng", return_tensors="pt")
    output_tokens = model.generate(**text_inputs, tgt_lang="isl", generate_speech=False)
    return processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)


task = "ABSA"
dataset = "asqp_rest15"

# Load the dataset with pandas
df = pd.read_csv(f"data/{task}/english/{dataset}/test.csv")

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Open output CSV file

with open(f"data/{task}/icelandic/{dataset}/test.csv", "w+", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(["original_id", "utterance", "label"])

    # Iterate over the dataset and translate each row
    for index, row in df.iterrows():
        comment = row['text']
        translated_comment = translate_text(comment)


        label_arr = ast.literal_eval(row['label_text'])
        translated_labels = []
        for overall_aspect, aspect, description, label in label_arr:
            labels = []
            for to_translate in [overall_aspect, aspect, description]:
                if to_translate != "NULL":
                    translated_val = translate_text(to_translate)
                else:
                    translated_val = to_translate

                labels.append(translated_val)
            labels.append(label)

            translated_labels.append(tuple(labels))

        writer.writerow([row['original_index'], translated_comment, repr(translated_labels)])

print("Translation completed and saved to CSV.")