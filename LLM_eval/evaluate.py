import argparse
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast
import re
from anthropic import InternalServerError, RateLimitError
from openai import APITimeoutError, APIConnectionError
from tenacity import RetryError

from data.senticnet_is.senticnet import senticnet as senticnet_is
from data.senticnet_en.senticnet import senticnet as senticnet_en
from obj_models.gpt import GPTClient
from obj_models.senticnet import eval as senticnet_eval
from obj_models.anthropic import AnthropicClient
from obj_models.llama import LlamaCppClient
from obj_models.mistral import MistralAPIClient

from baseline import main as baseline_eval
from baseline_ABSA import main as baseline_absa_eval

from funcs import get_dataset_obj, process_tuple_f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--shot", type=int, default=0, help="")
    parser.add_argument("--task", type=str, default="MAST", help="Task to perform (SA, ABSA, MAST)")
    parser.add_argument("--lang", type=str, default="icelandic", help="Language to evaluate (english/icelandic)")
    parser.add_argument("--dataset", type=str, default="", help="[chat]")
    parser.add_argument("--models", type=str, default="models.txt", help="[chat]")
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--post_eval", type=bool, default=False)
    return parser.parse_args()


def parse_and_validate_args():
    args = parse_args()

    assert args.task.lower() in ["sa", "absa", "mast"], "Task is not supported"
    assert args.lang.lower() in ["icelandic", "english"], "Language is not supported"

    return args


def load_env():
    """Helper function to read environment variables - where API keys are stored"""

    with open(".env", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value


def get_preds(dataset, lang, model_name, task):

    if lang == "icelandic" and model_name == "senticnet":
        return senticnet_eval(senticnet_is, dataset)
    elif lang == "english" and model_name == "senticnet":
        return senticnet_eval(senticnet_en, dataset)
    
    if "gpt" in model_name:
        gpt = GPTClient(model_name)
        return gpt.evaluate(dataset, task)
    
    if model_name[:6].lower() == "claude":
        anthropic = AnthropicClient(model_name)
        return anthropic.evaluate(dataset, task)
    
    if "mistral" in model_name.lower():
        mistral = MistralAPIClient(model_name)
        return mistral.evaluate(dataset, task)
    
    if "llama" in model_name.lower():
        llama = LlamaCppClient(model_name)
        return llama.evaluate(dataset, task)
    
    if "gemma" in model_name.lower():
        gemma = LlamaCppClient(model_name)
        return gemma.evaluate(dataset, task)


def calculate_metrics(labels, preds, task, dataset_name, shot, model_name, class_labels):
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, labels=class_labels, average=None)
    rec = recall_score(labels, preds, labels=class_labels, average=None)
    f1 = f1_score(labels, preds, labels=class_labels, average=None)

    avg_pre = precision_score(labels, preds, labels=class_labels, average='micro')
    avg_rec = recall_score(labels, preds, labels=class_labels, average='micro')
    avg_f1 = f1_score(labels, preds, labels=class_labels, average='micro')

    with open("results/" + task + "/" + dataset_name + f"/results_{shot}.txt", "a") as file:
        for ind, class_name in enumerate(class_labels):
            file.write(f"{model_name}, {class_name}, {acc:.2f}, {pre[ind]:.2f}, {rec[ind]:.2f}, {f1[ind]:.2f}, {datetime.now().strftime('%D-%H:%M:%S')}\n")
        
        file.write(f"{model_name}, AVERAGE, {acc:.2f}, {avg_pre:.2f}, {avg_rec:.2f}, {avg_f1:.2f}, {datetime.now().strftime('%D-%H:%M:%S')}\n")

    

def eval(model_name, args, dataset):
    print(f"Evaluation of {model_name} starting")

    if model_name == "electra":
        baseline_eval(task=args.task, dataset=dataset.name)

    else:
        labels, preds = get_preds(dataset, args.lang, model_name, args.task)

        calculate_metrics(labels, preds, args.task, dataset.name, args.shot, model_name, dataset.labels)


def str_to_tuple(text_arr):
    text_arr = text_arr.lower().strip()
    text_arr.replace("’", "'")
    match = re.search(r'\[.*?\]', text_arr)
    if match:
        content = match.group(0)
        try:
            content_list = ast.literal_eval(content)
        except:
            print("Error parsing content: ", content)
            content_list = []
        
        return content_list
    else:
        return []




def eval_ABSA(model_name, args, dataset):
    print(f"Evaluation of {model_name} starting")

    if model_name == "electra":
        baseline_absa_eval(dataset=dataset.name)
    
    else:

        labels, preds = get_preds(dataset, args.lang, model_name, args.task)

        precision, recall, micro_f1 = process_tuple_f1(labels, preds)

        with open("results/" + args.task + "/" + dataset.name + f"/results_{args.shot}.txt", "a") as file:
            file.write(f"{model_name}, {precision:.2f}, {recall:.2f}, {micro_f1:.2f}, {datetime.now().strftime('%D-%H:%M:%S')}\n")


def parse_models(models_file):
    models = []
    with open(models_file, "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                models.append(line.strip())
    
    return np.asarray(models)


def main():
    args = parse_and_validate_args()

    # read env keys
    load_env()
    models_file = args.models
    models = parse_models(models_file)
    print(models)

    datasets = list()

    if args.dataset == "" and args.task:
        if args.task == "SA":
            datasets += ["asc_rest14", "RU_movie_reviews", "ice_and_fire_SA"]
        elif args.task == "ABSA":
            datasets += ["uabsa_rest16", "aste_rest14", "asqp_rest15"]
        elif args.task == "MAST":
            datasets += ["compsent19", "implicit", "stance", "ice_and_fire_hate", 
                "ice_and_fire_irony", "ice_and_fire_offensive", "ice_and_fire_ER"]
        else:
            print("Invalid options")
    else:
        datasets.append(args.dataset)
        

    for dataset in datasets:

            print("Evaluating ", dataset)

            dataset_path = "data/" + args.task + "/" + args.lang + "/" + dataset

            dataset_obj = get_dataset_obj(dataset, dataset_path, args.shot)

            if args.task == "ABSA":
                if not os.path.isfile("results/" + args.task + "/" + dataset + f"/results_{args.shot}.txt"):
                    with open("results/" + args.task + "/" + dataset + f"/results_{args.shot}.txt", "w") as file:
                        file.write("Model,Precision, Recall, Micro-F1, timestamp\n")
                
                for model_name in models:
                    try:
                        eval_ABSA(model_name, args, dataset_obj)
                    except (InternalServerError, RateLimitError, APITimeoutError) as e:
                        print(f"An error occurred during evaluation of {model_name}")
                        print(e)
            else:
                if not os.path.isfile("results/" + args.task + "/" + dataset + f"/results_{args.shot}.txt"):
                    with open("results/" + args.task + "/" + dataset + f"/results_{args.shot}.txt", "w") as file:
                        file.write("Model, Class, Accuracy, Precision, Recall, F1, timestamp\n")
                
                for model_name in models:
                    try:
                        eval(model_name, args, dataset_obj)
        
                    except (InternalServerError, RateLimitError, APITimeoutError, APIConnectionError, RetryError) as e:
                        print(f"\tAn error occurred during evaluation of {model_name}")
                        print(e)



if __name__ == "__main__":
    main()