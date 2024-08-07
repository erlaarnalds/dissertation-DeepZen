


import random
from evaluate import get_dataset_obj
import csv

seed = 38
random.seed(seed)

dataset_name = "imdb_google"
dataset_path = "data/SA/icelandic/"
dataset = get_dataset_obj(dataset_name, dataset_path)

shots = 500

# Open file in write mode and write the header
with open(dataset_path + "/test.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["original_id", "utterance", "label"])

inds = []

# Generate random samples
for i in range(shots):
    random_no = int(random.uniform(0, len(dataset)))

    while random_no in inds:
        random_no = int(random.uniform(0, len(dataset)))

    inds.append(random_no)

    # Open file in append mode and write the row
    with open(dataset_path + "/test.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([random_no, dataset.dataset.iloc[random_no]['review'], dataset.dataset.iloc[random_no]['sentiment']])

