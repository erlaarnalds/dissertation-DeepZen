import argparse
from datasets import Dataset
import numpy as np
from senticnet_is.senticnet import senticnet as senticnet_is
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
import os



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset", type=str, default="datasets/SA/icelandic/ice_and_fire_SA", help="[chat]")
    parser.add_argument("--baseline", type=bool, default=True, help="Should baseline be included")
    #parser.add_argument("--models", type=str, default="gpt4o", help="List models to be evaluated in a string")
    return parser.parse_args()


def load_env():
    """Helper function to read environment variables - where API keys are stored"""

    with open(".env", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value


def preprocess(sentence):
    # switch out icelandic chars as they are not present in senticnet
    ice_to_eng = {
    'á': 'a', 'ð': 'd', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ý': 'y', 'þ': 'th', 'æ': 'ae', 'ö': 'o',
    'Á': 'A', 'Ð': 'D', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ý': 'Y', 'Þ': 'Th', 'Æ': 'Ae', 'Ö': 'O'
    }
    for ice_char, eng_char in ice_to_eng.items():
        sentence = sentence.replace(ice_char, eng_char) 

    # remove anything that is not a word character or white space
    words = re.sub(r'[^\w\s]', '', sentence).lower().split()

    return words

def parse_label(label):
    classes = {'Neikvætt': -1.0, 'Jákvætt': 1.0, 'Hlutlaust': 0.0}

    label_arr = label.split(";")

    scores = []

    for label in label_arr:
        scores.append(classes[label])
    
    return sum(scores) / len(label_arr)


def sign(pred):
    threshold = 0.5

    if pred > threshold:
        return "Positive"
    if pred < (threshold-1):
        return "Negative"
    else:
        return "Neutral"




def senticnet(senticnet, dataset):

    labels = []
    predictions = []
    
    for line in dataset:
        sentiment_scores = []
        comment = line['comment_body']
        words = preprocess(comment)

        for word in words:
            if word in senticnet:
                # get polarity value of word
                sentiment_scores.append(senticnet[word][7])
    
        if len(sentiment_scores) > 0:
            score = sum(sentiment_scores) / len(sentiment_scores)
        else:
            score = 0.0
        predictions.append(sign(score))
        true_val = parse_label(line['label'])
        labels.append(sign(true_val))


    return labels, predictions


def rank_utterance(client, system_msg, utterance):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": system_msg},
          {"role": "user", "content": utterance}
        ]
    )

    return response.choices[0].message.content



def gpt_eval(dataset):
    client = OpenAI()
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed.
    """

    labels = []
    predictions = []

    for line in dataset:
        sentiment_scores = []
        comment = line['comment_body']

        pred = rank_utterance(client, system_msg, comment)
        predictions.append(pred)

        true_val = parse_label(line['label'])
        labels.append(sign(true_val))
    
    return labels, predictions



def main():
    args = parse_args()
    classes = ["Positive", "Negative", "Neutral"]

    # read env keys
    load_env()

    # Load SenticNet data

    models = ["gpt4o"]
    dataset_path = args.dataset

    dataset = Dataset.load_from_disk(dataset_path)

    # establish baseline performance with senticnet
    sentic_labels, sentic_preds = senticnet(senticnet_is, dataset)

    acc = accuracy_score(sentic_labels, sentic_preds)
    pre = precision_score(sentic_labels, sentic_preds, labels=classes, average=None)
    rec = recall_score(sentic_labels, sentic_preds, labels=classes, average=None)
    f1 = f1_score(sentic_labels, sentic_preds, labels=classes, average=None)

    with open("results/ice_and_fire_SA", "w") as file:
        file.write("Model, Class, Accuracy, Precision, Recall, F1\n")
        
        model = "SenticNet"

        for ind, class_name in enumerate(classes):
            file.write(f"{model}, {class_name}, {acc:.2f}, {pre[ind]:.2f}, {rec[ind]:.2f}, {f1[ind]:.2f}\n")


    # evaluate with gpt
    print("GPT evaluation starting...")
    gpt_labels, gpt_predictions = gpt_eval(dataset)

    acc = accuracy_score(gpt_labels, gpt_predictions)
    pre = precision_score(gpt_labels, gpt_predictions, labels=classes, average=None)
    rec = recall_score(gpt_labels, gpt_predictions, labels=classes, average=None)
    f1 = f1_score(gpt_labels, gpt_predictions, labels=classes, average=None)


    with open("results/ice_and_fire_SA", "a") as file:
        file.write("Model, Class, Accuracy, Precision, Recall, F1\n")
        
        model = "GPT-4o"

        for ind, class_name in enumerate(classes):
            file.write(f"{model}, {class_name}, {acc:.2f}, {pre[ind]:.2f}, {rec[ind]:.2f}, {f1[ind]:.2f}\n")



if __name__ == "__main__":
    main()