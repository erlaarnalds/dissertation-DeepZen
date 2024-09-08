from evaluate import get_dataset_obj
import argparse
import random
import csv
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ABSA", help="Task to perform (SA, ABSA, MAST)")
    parser.add_argument("--dataset", type=str, default="ubasa_rest16", help="Dataset to set up")
    return parser.parse_args()

args = parse_args()


seed = 38
random.seed(seed)

dataset_name = args.dataset
task = args.task
dataset_path = f"data/{task}/icelandic/{dataset_name}"
dataset_obj = get_dataset_obj(dataset_name=dataset_name, dataset_path=dataset_path, shot=0)

shots = 5



with open(dataset_path + f"/few_shot_{shots}.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["utterance", "label"])

if "ice_and_fire" in dataset_name:
    dataset_obj.dataset["final_label"] = dataset_obj.dataset.apply(lambda row: dataset_obj.get_label(dataset_obj.parse_label(row['label'], dataset_obj.classes)), axis=1)

if task == "ABSA":
    dataset_obj.dataset['label'] = dataset_obj.dataset['label'].apply(ast.literal_eval)

inds = []


for class_inst in list(dataset_obj.labels):
    print(class_inst)

    def has_label(label_list):
        return any(label[-1] == class_inst for label in label_list)

    if "ice_and_fire" in dataset_name:
        filtered_set = dataset_obj.dataset[dataset_obj.dataset['final_label'] == class_inst].reset_index(drop=True)

    elif task == "ABSA":
        filtered_set = dataset_obj.dataset[dataset_obj.dataset['label'].apply(has_label)].reset_index(drop=True)
    else:
        filtered_set = dataset_obj.dataset[dataset_obj.dataset['label'] == class_inst].reset_index(drop=True)


    print(len(filtered_set))
    for i in range(shots):
        random_no = int(random.uniform(0, len(filtered_set)))
        ind = filtered_set.iloc[random_no]['original_id']
        tries = 0
        while ind in inds and len(filtered_set) > 0 and tries < 5:
            random_no = int(random.uniform(0, len(filtered_set)))
            ind = filtered_set.iloc[random_no]['original_id']
            tries += 1

        print(random_no)
        inds.append(ind)

        row = filtered_set.iloc[random_no]

        with open(dataset_path + f"/few_shot_{shots}.csv", "a+") as file:
            writer = csv.writer(file)
            writer.writerow([row['utterance'], row['aspect'], row['label']])


with open(dataset_path + f"/train_ids.csv", "w+") as file:
    writer = csv.writer(file)
    writer.writerow(["ids"])
    for ind in inds:
        writer.writerow([ind])