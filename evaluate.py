import argparse
from datasets import Dataset
import numpy as np
from datasets_local.senticnet_is.senticnet import senticnet as senticnet_is
from datasets_local.senticnet_en.senticnet import senticnet as senticnet_en
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from anthropic import Anthropic
import os



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--task", type=str, default="SA", help="Task to perform (SA, ABSA, M...)")
    parser.add_argument("--lang", type=str, default="icelandic", help="Language to evaluate (english/icelandic)")
    parser.add_argument("--dataset", type=str, default="ice_and_fire_SA", help="[chat]")
    parser.add_argument("--models", type=str, default="models.txt", help="[chat]")
    parser.add_argument("--baseline", type=str, default="senticnet")
    return parser.parse_args()


def parse_and_validate_args():
    args = parse_args()

    assert args.task.lower() in ["sa"]
    assert args.lang.lower() in ["icelandic", "english"]

    return args


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


def senticnet_eval(senticnet, dataset):

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


def rank_gpt_utterance(client, model_name, system_msg, utterance):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
          {"role": "system", "content": system_msg},
          {"role": "user", "content": utterance}
        ]
    )

    return response.choices[0].message.content


def rank_anthropic_utterance(client, model_name, system_msg, utterance):

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        system=system_msg,
        messages=[
          {"role": "user", 
           "content": [
               {
                   "type": "text",
                   "text": utterance
               }
           ]}
        ]
    )

    return response.content[0].text


def anthropic_eval(dataset, model_name):
    client = Anthropic()
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """

    labels = []
    predictions = []

    for i, line in enumerate(dataset):
        print(i)
        comment = line['comment_body']

        if not comment.isspace() and comment!="" and comment != None:
            pred = rank_anthropic_utterance(client, model_name, system_msg, comment)
            predictions.append(pred)

            true_val = parse_label(line['label'])
            labels.append(sign(true_val))
    
    return labels, predictions


def gpt_eval(dataset, model_name):
    client = OpenAI()
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """

    labels = []
    predictions = []

    for line in dataset:
        comment = line['comment_body']

        pred = rank_gpt_utterance(client, model_name, system_msg, comment)
        predictions.append(pred)

        true_val = parse_label(line['label'])
        labels.append(sign(true_val))
    
    return labels, predictions


def eval(model_name, lang, dataset, classes):
    print(f"Evaluation of {model_name} starting")

    labels = np.zeros(2)
    preds = np.zeros(2)

    if lang == "icelandic" and model_name == "senticnet":
        labels, preds = senticnet_eval(senticnet_is, dataset)
    elif lang == "english" and model_name == "senticnet":
        labels, preds = senticnet_eval(senticnet_en, dataset)
    
    if model_name[:3] == "gpt":
        labels, preds = gpt_eval(dataset, model_name)
    
    if model_name[:6] == "claude":
        labels, preds = anthropic_eval(dataset, model_name)

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, labels=classes, average=None)
    rec = recall_score(labels, preds, labels=classes, average=None)
    f1 = f1_score(labels, preds, labels=classes, average=None)

    avg_pre = precision_score(labels, preds, labels=classes, average='micro')
    avg_rec = recall_score(labels, preds, labels=classes, average='micro')
    avg_f1 = f1_score(labels, preds, labels=classes, average='micro')

    with open("results/ice_and_fire_SA", "a") as file:
        for ind, class_name in enumerate(classes):
            file.write(f"{model_name}, {class_name}, {acc:.2f}, {pre[ind]:.2f}, {rec[ind]:.2f}, {f1[ind]:.2f}\n")
        
        file.write(f"{model_name}, AVERAGE, {acc:.2f}, {avg_pre:.2f}, {avg_rec:.2f}, {avg_f1:.2f}\n")


def parse_models(models_file):
    models = []
    with open("models.txt", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                models.append(line.strip())
    
    return np.asarray(models)



def main():
    args = parse_and_validate_args()
    classes = ["Positive", "Negative", "Neutral"]

    # read env keys
    load_env()
    models_file = args.models
    models = parse_models(models_file)
    print(models)
    dataset_path = "datasets_local/" + args.task + "/" + args.lang + "/" + args.dataset

    dataset = Dataset.load_from_disk(dataset_path)
    
    with open("results/ice_and_fire_SA", "w") as file:
        file.write("Model, Class, Accuracy, Precision, Recall, F1\n")

    for model_name in models:
        eval(model_name, args.lang, dataset, classes)


if __name__ == "__main__":
    main()