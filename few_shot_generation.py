from evaluate import get_dataset_obj
import random
import csv


seed = 38
random.seed(seed)

dataset_name = "RU_movie_reviews"
dataset_path = "data/SA/icelandic/" + dataset_name
dataset_obj = get_dataset_obj(dataset_name=dataset_name, dataset_path=dataset_path, shot=0)

shots = 5

with open(dataset_path + f"/few_shot_{shots}.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["utterance", "label"])

for class_inst in list(dataset_obj.classes.keys()):
    inds = []
    print(class_inst)
    filtered_set = dataset_obj.dataset[dataset_obj.dataset['label'] == class_inst].reset_index(drop=True)
    for i in range(shots):
        random_no = int(random.uniform(0, len(filtered_set)))
        while random_no in inds:
            random_no = int(random.uniform(0, len(filtered_set)))

        print(random_no)

        row = filtered_set.iloc[random_no]

        with open(dataset_path + f"/few_shot_{shots}.csv", "a+") as file:
            writer = csv.writer(file)
            writer.writerow([row['utterance'], row['label']])