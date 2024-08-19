Add API keys to a .env file, using the following template:

```
OPENAI_KEY = ...
````



Finetuning GPT:
python finetuning/gpt.py --create file --dataset asc_rest14
copy id
python finetuning/gpt.py --create model --id <file_id> --dataset aste_rest14
copy id to check status
python finetuning/gpt.py --check model --id <id>