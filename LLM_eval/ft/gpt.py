import argparse
from openai import OpenAI
import os

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

    client = OpenAI()
    if args.create == "file":
        resp = client.files.create(
            file=open(f"data/{task}/icelandic/{dataset_name}/finetuning.jsonl", "rb"),
            purpose="fine-tune"
        )
        print(resp.id)

    model_name = "gpt-4o-2024-08-06"
    #model_name = "gpt-4o-mini-2024-07-18"

    if args.create == "model":
        resp = client.fine_tuning.jobs.create(
            training_file=args.id, 
            model=model_name,
            suffix=dataset_name
        )
        print(resp.id)

    elif args.check == "file":
        resp = client.files.retrieve(args.id)
        print(resp.id)
    
    elif args.check == "model":
        resp = client.fine_tuning.jobs.retrieve(args.id)
        print("Status of job: ", resp.status)

    if args.delete == "model" and args.id == "":
        model_ids = [
            "ft:gpt-4o-mini-2024-07-18:personal:asqp-rest15:9z1bTnsl:ckpt-step-60",
            "ft:gpt-4o-mini-2024-07-18:personal:asqp-rest15:9z1bTS6p:ckpt-step-75",
            "ft:gpt-4o-mini-2024-07-18:personal:asqp-rest15:9z1bT1vj",
            "ft:gpt-4o-mini-2024-07-18:personal:uabsa-rest16:9wcmFQkC",
            "ft:gpt-4o-mini-2024-07-18:personal:aste-rest14:9wcu2WWV",
            "ft:gpt-4o-2024-08-06:personal:aste-rest14:9yzMAZxS:ckpt-step-75",
            "ft:gpt-4o-2024-08-06:personal:aste-rest14:9yzMAiRH:ckpt-step-60",
            "ft:gpt-4o-2024-08-06:personal:aste-rest14:9yzMASK7",
            "ft:gpt-4o-2024-08-06:personal:uabsa-rest16:9zQmgp3t:ckpt-step-90",
            "ft:gpt-4o-2024-08-06:personal:uabsa-rest16:9zQmg9RH",
            "ft:gpt-4o-2024-08-06:personal:asqp-rest15:9zQpPj1f:ckpt-step-75",
            "ft:gpt-4o-2024-08-06:personal:uabsa-rest16:9zQmgMPs:ckpt-step-45",
            "ft:gpt-4o-2024-08-06:personal:asqp-rest15:9zQpPMMq:ckpt-step-60",
            "ft:gpt-4o-2024-08-06:personal:asqp-rest15:9zQpQWIr",
            "ft:gpt-4o-2024-08-06:personal:compsent19:9zRhPon1",
            "ft:gpt-4o-2024-08-06:personal:compsent19:9zRhOVWS:ckpt-step-80",
            "ft:gpt-4o-2024-08-06:personal:compsent19:9zRhO7KB:ckpt-step-90",
        ]

        for model_id in model_ids:
            resp = client.models.delete(model_id)
            print(resp)
    
    elif args.delete == "model" and args.id != "":
        resp = client.models.delete(args.id)
        print(resp)

    elif args.delete == "file" and args.id != "":
        resp = client.files.delete(args.id)
        print(resp)

    if args.list:
        resp = client.models.list()
        
        for obj in resp.data:
            print(obj.id)