from evaluate import get_dataset_obj
import random
import json
import ast
import pandas as pd


seed = 38
random.seed(seed)

dataset_name = "ice_and_fire_offensive"
task = "MAST"
dataset_path = f"data/{task}/icelandic/{dataset_name}"
dataset_obj = get_dataset_obj(dataset_name=dataset_name, dataset_path=dataset_path, shot=0)

test_lines = pd.read_csv(dataset_path + "/few_shot_5.csv")

system_msg = dataset_obj.get_system_msg()
system_msg = system_msg.replace("        ", "")

gpt = False
mistral = True


for i, row in test_lines.iterrows():

    utterance = f"Snippet: {row['utterance']}\n"
    label = row['label']

    if gpt:
        messages = []
        messages.append({"role":"system","content":system_msg})
        messages.append({"role":"user","content":utterance})
        messages.append({"role":"assistant","content":label})
        
        data = {"messages":messages}


        # Writing to a JSONL file
        with open(dataset_path + '/finetuning.jsonl', 'a') as file:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + "\n")
    
    if mistral:

        input_text = "[INST]" + system_msg + "[/INST]\n" + utterance

        df_formatted = {
                "messages": [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": label},
                ]
            }
        

        # Writing to a JSONL file
        with open(dataset_path + '/finetuning_mistral.jsonl', 'a') as file:
            json_line = json.dumps(df_formatted, ensure_ascii=False)
            file.write(json_line + "\n")





