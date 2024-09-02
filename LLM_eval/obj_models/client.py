import os
import csv

class Client():

    def __init__(self, model_name, finetuned):
        self.model_name = model_name
        self.finetuned = finetuned


    def rank_utterance(self):
        pass
    

    def evaluate(self, dataset, task=""):

        system_msg = dataset.get_system_msg()

        labels = []
        predictions = []

        for ind, utterance, label in dataset:

            if not self.finetuned or (self.finetuned and ind not in dataset.train_ids):

                pred = self.rank_utterance(system_msg, utterance)
                predictions.append(pred)

                if task == "ABSA":
                    true_label = label
                else:
                    true_val = dataset.parse_label(label, dataset.classes)
                    true_label = dataset.get_label(true_val)
                
                labels.append(true_label)

                if not os.path.isfile(f"logs/{dataset.name}/{self.model_name}_shot_{dataset.shot}.csv"):
                    with open(f"logs/{dataset.name}/{self.model_name}_shot_{dataset.shot}.csv", "w") as file:
                        writer = csv.writer(file)
                        writer.writerow(["utterance", "prediction" ,"label"])

                with open(f"logs/{dataset.name}/{self.model_name}_shot_{dataset.shot}.csv", "a") as file:
                    writer = csv.writer(file)
                    writer.writerow([utterance, pred, true_label])

        return labels, predictions
