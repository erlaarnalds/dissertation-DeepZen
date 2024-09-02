from evaluate import get_dataset_obj
import random
import csv
import ast


seed = 38
random.seed(seed)

dataset_name = "ubasa_rest16"
task = "ABSA"
dataset_path = f"data/{task}/icelandic/{dataset_name}"
dataset_obj = get_dataset_obj(dataset_name=dataset_name, dataset_path=dataset_path, shot=0)

shots = 5

with open(dataset_path + f"/few_shot_{shots}.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["utterance", "label"])


# for iac
#dataset_obj.dataset["final_label"] = dataset_obj.dataset.apply(lambda row: dataset_obj.get_label(dataset_obj.parse_label(row['label'], dataset_obj.classes)), axis=1)

# for absa
dataset_obj.dataset['label'] = dataset_obj.dataset['label'].apply(ast.literal_eval)

inds = []
for class_inst in list(dataset_obj.classes.keys()):
#for class_inst in list(dataset_obj.labels):
    print(class_inst)

    def has_label(label_list):
        return any(label == class_inst for _, _, _, label in label_list)

    #filtered_set = dataset_obj.dataset[dataset_obj.dataset['final_label'] == class_inst].reset_index(drop=True)
    #filtered_set = dataset_obj.dataset[dataset_obj.dataset['label'] == class_inst].reset_index(drop=True)
    filtered_set = dataset_obj.dataset[dataset_obj.dataset['label'].apply(has_label)].reset_index(drop=True)
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