


import random
from evaluate import get_dataset_obj
import csv
import pandas as pd
from evaluate import calculate_metrics

seed = 38
random.seed(seed)

dataset_name = "ice_and_fire_irony"
task = "MAST"
shot = 5
model = "gemma-2"
dataset_path = f"intermediate_res/{dataset_name}_shot_{shot}_{model}.csv"
dataset = pd.read_csv(dataset_path)

labels = dataset["label"].tolist()
preds = dataset["prediction"].tolist()

classes = ['irony', 'not irony', 'unclear']

calculate_metrics(labels, preds, task, dataset_name, shot, model, classes)
