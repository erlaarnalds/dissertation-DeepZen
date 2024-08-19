import argparse
from openai import OpenAI
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", type=str, default="")
    parser.add_argument("--check", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--task", type=str, default="MAST")
    parser.add_argument("--id", type=str, default="file-FxmTgBxOnk0KxxZRqzQkhffn")
    return parser.parse_args()

def load_env():
    """Helper function to read environment variables - where API keys are stored"""

    with open(".env", "r") as file_stream:
        for line in file_stream:
            if not line.startswith('#') and line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value


if __name__ == "__main__":
    load_env()
    args = parse_args()

    dataset_name = args.dataset
    task = args.task

    client = OpenAI()
    if args.create == "file":
        resp = client.files.create(
            file=open(f"data/{task}/icelandic/{dataset_name}/finetuning.jsonl", "rb"),
            purpose="fine-tune"
        )
        print(resp.id)

    if args.create == "model":
        resp = client.fine_tuning.jobs.create(
            training_file=args.id, 
            model="gpt-4o-mini-2024-07-18",
            suffix=dataset_name
        )
        print(resp.id)

    elif args.check == "file":
        resp = client.files.retrieve(args.id)
        print(resp.id)
    
    elif args.check == "model":
        resp = client.fine_tuning.jobs.retrieve(args.id)
        print("Status of job: ", resp.status)