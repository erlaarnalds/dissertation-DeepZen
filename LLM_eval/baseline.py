from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer
import numpy as np

import html
import pandas as pd
import tokenizer
import re
import math
from evaluate import get_dataset_obj
import argparse
from datetime import datetime

stop_flag = 0


class TextCleaner:
    def __init__(self):
        pass

    def clean_html(self, txt):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", txt)

    def remove_brackets(self, txt):
        return re.sub(r"[()\[\]{}<>]", "", txt)

    def lower_case(self, txt):
        return txt.lower()

    def fix_repeated_characters(self, txt):
        return re.sub(r"(.)\1{5,}", r"\1", txt)

    def remove_overly_long_words(self, txt):
        return " ".join([t for t in txt.split(" ") if len(t) < 30])

    def remove_special_characters(self, txt):
        # remove special characters and digits except icelandic characters

        pattern = r"[^a-zA-záðéíóúýþæö.?!;:,\s]"
        txt = re.sub(pattern, "", txt)
        return txt

    def remove_noise(self, txt):

        if type(txt) == float and math.isnan(txt):
            txt = " "

        txt = html.unescape(txt)
        txt = self.clean_html(txt)
        txt = self.remove_brackets(txt)
        txt = self.lower_case(txt)
        txt = self.remove_special_characters(txt)
        txt = self.fix_repeated_characters(txt)
        txt = self.remove_overly_long_words(txt)
        return txt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MAST", help="Task to perform (SA, ABSA, MAST)")
    parser.add_argument("--dataset", type=str, default="", help="[chat]")
    return parser.parse_args()

RANDOM_SEED = 42
EPOCHS = 4
LEARNING_RATE = 1e-6
BATCH_SIZE = 4


model_name = "jonfd/electra-base-igc-is"
model_save_dir = "electra"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def tokenize_data(data, tokenizer, max_len=512):
    print(len(data))
    return tokenizer(
        data.tolist(), padding="max_length", truncation=True, max_length=max_len
    )

args = parse_args()
dataset = args.dataset
task = args.task
dataset_path = f"data/{task}/icelandic/{dataset}"
dataset_obj = get_dataset_obj(dataset, dataset_path, 0)


df = pd.DataFrame(columns=['snippet', 'label'])


for original_id, sentence, label in dataset_obj:
    ind = dataset_obj.parse_label(label, dataset_obj.classes)
    eng_label = dataset_obj.get_label(ind)
    print(eng_label)
    label_int = dataset_obj.labels.index(eng_label)
    print(label_int)
    df = df._append({'snippet': sentence, 'label': int(label_int)}, ignore_index=True)


cleaner = TextCleaner()

df["snippet"] = df.snippet.apply(cleaner.remove_noise)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


X_train, X_temp, y_train, y_temp = train_test_split(
    df["snippet"], df["label"], test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return {
        "acc": accuracy_score(labels, predictions),
        "per": precision_score(labels, predictions, average="micro"),
        "rec": recall_score(labels, predictions, average="micro"),
        "f1": f1_score(labels, predictions, average="micro"),
    }


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(dataset_obj.labels),
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# Tokenize the training data
train_data = tokenize_data(X_train, tokenizer)

# Tokenize the validation data
val_data = tokenize_data(X_val, tokenizer)

# Tokenize the test data
test_data = tokenize_data(X_test, tokenizer)
print(X_train.iloc[0])
print(y_train.iloc[0])
train_dataset = SentimentDataset(train_data, y_train)
val_dataset = SentimentDataset(val_data, y_val)
test_dataset = SentimentDataset(test_data, y_test)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train_dataset) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_training_steps=total_steps, num_warmup_steps=0
)

log_dir = "logs"

training_args = TrainingArguments(
    output_dir=f"baseline_checkpoints/results/{model_save_dir}",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=log_dir,
    load_best_model_at_end=True,
    learning_rate=LEARNING_RATE,
)

# Make all tensors contiguous before saving
for name, param in model.named_parameters():
    if not param.is_contiguous():
        param.data = param.contiguous()
        
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
)


trainer.train()

results = trainer.evaluate(test_dataset)
with open("results/" + task + "/" + dataset + f"/results_0.txt", "a") as file:
        file.write(f"BASELINE, AVERAGE, {results['eval_acc']:.2f}, {results['eval_per']:.2f}, {results['eval_rec']:.2f}, {results['eval_f1']:.2f}, {datetime.now().strftime('%D-%H:%M:%S')}\n")
print("test results:", results)

model.save_pretrained(f"baseline_checkpoints/{model_save_dir}")
tokenizer.save_pretrained(f"baseline_checkpoints/{model_save_dir}")