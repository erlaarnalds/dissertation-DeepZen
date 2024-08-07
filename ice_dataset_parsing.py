# huggingface-cli login
# pip install datasets
from datasets import load_dataset, Dataset
from bs4 import BeautifulSoup
import numpy as np
import csv


def clean_html(text):
    html_parser = BeautifulSoup(text, 'html.parser')
    paragraphs = html_parser.find_all('p')

    cleaned_text = '\n'.join(p.get_text().strip() for p in paragraphs)
    cleaned_text = cleaned_text.replace('\xa0', ' ')

    return cleaned_text


def clean_comment_history(comments):
    if comments == None:
        return
    
    comments_arr = comments.split('||')

    cleaned_comments_arr = list()

    for comment in comments_arr:
        cleaned_comments_arr.append(clean_html(comment))
    
    return '||'.join(cleaned_comments_arr)



def clean(row, index):
    row['id'] = index
    row['blog_text'] = clean_html(row['blog_text'])
    row['comment_body'] = clean_html(row['comment_body'])
    row['previous_comments'] = clean_comment_history(row['previous_comments'])

    return row


def filter(dataset):
    merged_info = {}

    for row in dataset:
        comment = row['comment_body']

        if merged_info.get(comment) == None:
            merged_info[comment] = {'rating':[], 'data': row}
        
        merged_info[comment]['rating'].append(row['label'])
    
    merged_dataset = []
    for comment, line in merged_info.items():
        data = line['data']
        data['label'] = ";".join(line['rating'])
        merged_dataset.append(data)

    return merged_dataset


if __name__ == "__main__":

    ds = load_dataset("hafsteinn/ice_and_fire")
    # only train section is provided
    train_data = ds['train']

    # dataset is multi-task, so certain label is isolated
    task_train = train_data.filter(lambda ds: ds['task_type'] == 'Almennt viðhorf til aðalviðfangsefnisins')
    print(len(task_train))
    print(len(train_data))

    # Remove html from info
    cleaned_task_data = task_train.map(clean, with_indices=True)


    # Join together opinions from different reviewers of the same comment
    filtered_data = filter(cleaned_task_data)
    print(len(filtered_data))


    max_rows = 500

    if len(filtered_data) > max_rows:
        filtered_data = filtered_data[:max_rows]


    with open("data/MAST/icelandic/ice_and_fire_stance/test.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "utterance", "label"])
        writer.writeheader()

        for row in filtered_data:
            writer.writerow({"id":row['id'], "utterance": row['comment_body'], "label":row['label']})


