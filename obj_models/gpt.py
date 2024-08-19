from openai import OpenAI
import os
import csv
from obj_models.client import Client


class GPTClient(Client):
    def __init__(self, model_name, finetuned=False):
        self.client = OpenAI()
        if model_name[:2] == "ft":
            finetuned = True

        super().__init__(model_name, finetuned)

        

    def rank_utterance(self, system_msg, utterance):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": utterance}
            ]
        )

        return response.choices[0].message.content.lower()



# def eval(dataset, model_name, task="", finetuned=False):
#     client = OpenAI()

#     system_msg = dataset.get_system_msg()

#     labels = []
#     predictions = []

#     for ind, utterance, label in dataset:

#         if not finetuned or (finetuned and ind not in dataset.train_ids):

#             pred = rank_utterance(client, model_name, system_msg, utterance)
#             predictions.append(pred)

#             if task == "ABSA":
#                 true_label = label
#             else:
#                 true_val = dataset.parse_label(label, dataset.classes)
#                 true_label = dataset.get_label(true_val)
            
#             labels.append(true_label)

#             if not os.path.isfile(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv"):
#                 with open(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv", "w") as file:
#                     writer = csv.writer(file)
#                     writer.writerow(f"utterance,prediction,label\n")

#             with open(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv", "a") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([utterance, pred, true_label])

#     return labels, predictions