# LLM Evalution

This project contains the necessary code to evaluate a number of LLMs on Icelandic sentiment analysis.

The datasets are stored in the /data folder, divided by the evaluation task.

## Getting started

Begin by installing all package requirements:
```
pip install -r requirements.txt 
````

The models that can be evaluated are listed in *models.txt*. Uncomment the LLMs you wish to evaluate. The code assumes to find the relevant API keys in an environment file. This can be created like so:

```
touch .env
nano .env
````

Add the relevant API keys in the following format:

```
OPENAI_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
````

Keys can be created through the LLM API sites. 

## Creating fine-tuned models

To evaluate fine-tuned models, you first need to add the models to your account. You can do this using the evaluation code, as relevant files have already been created. 

### Add a file
Run the following code to add a file to your account. This will add the relevant datasets to your account. 

```
python finetuning/gpt.py --create file --task <task being evaluated> --dataset <dataset_name>
````

In this case, the task can be SA, ABSA or MAST. The terminal output will be the file id of the file being created. You will need this for later steps.

### Start model training

You will now use the file you added to train a model:

```
python finetuning/gpt.py --create model --id <file_id> --task <task being evaluated> --dataset <dataset_name>
````

The output will be the training id. Once the model has been fully trained, the model name will be created. You can now add this name to models.txt to evaluate the fine-tuned model on your dataset. 

### Check training status

Since training might take some time, you can check the training status with the following code:

```
python finetuning/gpt.py --check model --id <job_id>
````

If you wish to train a mistral model instead, simply run python finetuning/mistral.py instead of python finetuning/gpt.py 

### List model names

To list all available models, run:

```
python finetuning/gpt.py --list True
````
### Delete model

The models include a storage fee. If you wish to delete a model, you can run:

```
python finetuning/gpt.py --delete model --id <model_id>
````

Where the model id can be found by using the list command.


## Running the code

You can now run the evaluation code using the following command:

```
python evaluate.py --task <task_name> --dataset <dataset_name>
````

If you wish to evalute all datasets within a task, simply omit the dataset parameter. 
