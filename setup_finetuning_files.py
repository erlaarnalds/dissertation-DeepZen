from evaluate import get_dataset_obj
import random
import json
import ast
import pandas as pd


seed = 38
random.seed(seed)

dataset_name = "implicit"
task = "MAST"
dataset_path = f"data/{task}/icelandic/{dataset_name}"
dataset_obj = get_dataset_obj(dataset_name=dataset_name, dataset_path=dataset_path, shot=0)

test_lines = pd.read_csv(dataset_path + "/few_shot_5.csv")

system_msg = dataset_obj.get_system_msg()
system_msg = system_msg.replace("        ", "")

for i, row in test_lines.iterrows():

    messages = []
    messages.append({"role":"system","content":system_msg})
    messages.append({"role":"user","content":row['utterance']})
    messages.append({"role":"assistant","content":row['label']})
    
    data = {"messages":messages}


    # Writing to a JSONL file
    with open(dataset_path + '/finetuning.jsonl', 'a') as file:
        json_line = json.dumps(data, ensure_ascii=False)
        file.write(json_line + "\n")



