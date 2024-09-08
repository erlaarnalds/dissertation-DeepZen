import argparse
from mistralai import Mistral
import os

verbose = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", type=str, default="")
    parser.add_argument("--check", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--task", type=str, default="SA")
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--delete", type=str, default="")
    parser.add_argument("--list", type=bool, default=False)
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

    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])

    if args.create == "file":
        resp = client.files.upload(file={
            "file_name": "finetuning_mistral.jsonl",
            "content": open(f"data/{task}/icelandic/{dataset_name}/finetuning_mistral.jsonl", "rb"),
        })

        print(resp.id)

    #model_name = "mistral-large-latest"
    model_name = "open-mistral-nemo"

    if args.create == "model":
        created_jobs = client.fine_tuning.jobs.create(
            model=model_name, 
            training_files=[{"file_id": args.id, "weight": 1}],
            hyperparameters={
                "training_steps": 10,
                "learning_rate":0.0001
            },
            suffix=dataset_name,
            auto_start=False
        )
        print(created_jobs.id)
        print(created_jobs)

        # start a fine-tuning job
        #client.fine_tuning.jobs.start(job_id = created_jobs.id)

        
    
    elif args.check == "model":
        resp = client.fine_tuning.jobs.get(job_id = args.id)
        print("Status of job: ", resp.status)

        if verbose:
            for event in resp.events:
                print(event)
        print(resp)

    if args.delete == "model" and args.id == "":
        model_ids = []

        for model_id in model_ids:
            resp = client.models.delete(model_id=model_id)
            print(resp)
    

    elif args.delete == "file" and args.id != "":
        resp = client.models.delete(model_id=args.id)
        print(resp)

    if args.list:
        resp = client.fine_tuning.jobs.list()

        for obj in resp.data:
            print(obj.fine_tuned_model)