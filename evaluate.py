import argparse
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets_local.senticnet_is.senticnet import senticnet as senticnet_is
from datasets_local.senticnet_en.senticnet import senticnet as senticnet_en
from obj_models.gpt import eval as gpt_eval
from obj_models.senticnet import eval as senticnet_eval
from obj_models.anthropic import eval as anthropic_eval
from obj_models.llama import eval as llama_eval

from obj_models.datasets import IceandFire, IMDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--task", type=str, default="SA", help="Task to perform (SA, ABSA, M...)")
    parser.add_argument("--lang", type=str, default="icelandic", help="Language to evaluate (english/icelandic)")
    parser.add_argument("--dataset", type=str, default="imdb_google", help="[chat]")
    parser.add_argument("--models", type=str, default="models.txt", help="[chat]")
    parser.add_argument("--baseline", type=str, default="senticnet")
    parser.add_argument("--rewrite_log", type=bool, default=False)
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



def eval(model_name, lang, dataset, classes):
    print(f"Evaluation of {model_name} starting")

    labels = np.zeros(2)
    preds = np.zeros(2)

    if lang == "icelandic" and model_name == "senticnet":
        labels, preds = senticnet_eval(senticnet_is, dataset)
    elif lang == "english" and model_name == "senticnet":
        labels, preds = senticnet_eval(senticnet_en, dataset)
    
    if model_name[:3].lower() == "gpt":
        labels, preds = gpt_eval(dataset, model_name)
    
    if model_name[:6].lower() == "claude":
        labels, preds = anthropic_eval(dataset, model_name)

    if model_name[:4].lower() == "meta":
        labels, preds = llama_eval(dataset, model_name)
    

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, labels=classes, average=None)
    rec = recall_score(labels, preds, labels=classes, average=None)
    f1 = f1_score(labels, preds, labels=classes, average=None)

    avg_pre = precision_score(labels, preds, labels=classes, average='micro')
    avg_rec = recall_score(labels, preds, labels=classes, average='micro')
    avg_f1 = f1_score(labels, preds, labels=classes, average='micro')

    with open("results/" + dataset.name, "a") as file:
        for ind, class_name in enumerate(classes):
            file.write(f"{model_name}, {class_name}, {acc:.2f}, {pre[ind]:.2f}, {rec[ind]:.2f}, {f1[ind]:.2f}, {datetime.now().strftime("%D-%H:%M:%S")}\n")
        
        file.write(f"{model_name}, AVERAGE, {acc:.2f}, {avg_pre:.2f}, {avg_rec:.2f}, {avg_f1:.2f}, {datetime.now().strftime("%D-%H:%M:%S")}\n")


def parse_models(models_file):
    models = []
    with open(models_file, "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                models.append(line.strip())
    
    return np.asarray(models)


def get_dataset_obj(dataset_name, dataset_path):
    if dataset_name == "ice_and_fire_SA":
        return IceandFire(dataset_path, dataset_name)
    
    elif dataset_name == "imdb_google":
        return IMDB(dataset_path, dataset_name)


def main():
    args = parse_and_validate_args()
    classes = ["positive", "negative", "neutral"]

    # read env keys
    load_env()
    models_file = args.models
    models = parse_models(models_file)
    print(models)
    dataset_path = "datasets_local/" + args.task + "/" + args.lang + "/" + args.dataset

    dataset_obj = get_dataset_obj(args.dataset, dataset_path)

    if not os.path.isfile("results/" + args.dataset):
        with open("results/" + args.dataset, "w") as file:
            file.write("Model, Class, Accuracy, Precision, Recall, F1, timestamp\n")

    for model_name in models:
        eval(model_name, args.lang, dataset_obj, classes)


if __name__ == "__main__":
    main()