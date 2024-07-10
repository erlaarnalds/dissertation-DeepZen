# huggingface-cli login
# pip install datasets
from datasets import load_dataset
from bs4 import BeautifulSoup
import numpy as np


classes = np.asarray(['undrun', 'fyrirlitning', 'reiði', 'sorg', 'gleði', 'hlutlaust', 'hneykslun', 'andstyggð'])


def clean_html(text):
    html_parser = BeautifulSoup(text, 'html.parser')
    paragraphs = html_parser.find_all('p')

    cleaned_text = '\n'.join(p.get_text().strip() for p in paragraphs)

    return cleaned_text


def clean_comment_history(comments):
    if comments == None:
        return
    
    comments_arr = comments.split('||')

    cleaned_comments_arr = list()

    for comment in comments_arr:
        cleaned_comments_arr.append(clean_html(comment))
    
    return '||'.join(cleaned_comments_arr)


if __name__ == "__main__":

    ds = load_dataset("hafsteinn/ice_and_fire")
    # only train section is provided
    train_data = ds['train']

    # dataset is multi-task, so emotion label is isolated
    er_train = train_data.filter(lambda ds: ds['task_type'] == 'Tilfinning')

    # Remove html from info
    cleaned_data = er_train.map(lambda dataset: {
        'blog_text': clean_html(dataset['blog_text']),
        'comment_body': clean_html(dataset['comment_body']),
        'previous_comments': clean_comment_history(dataset['previous_comments'])
    })

    cleaned_data.save_to_disk('../datasets/ERC/icelandic/ice_and_fire_ER')

    # repeat for Sentiment Analysis data
    # dataset is multi-task, so sentiment analysis label is isolated
    sa_train = train_data.filter(lambda ds: ds['task_type'] == 'Lyndi')

    cleaned_sa_data = sa_train.map(lambda dataset: {
        'blog_text': clean_html(dataset['blog_text']),
        'comment_body': clean_html(dataset['comment_body']),
        'previous_comments': clean_comment_history(dataset['previous_comments'])
    })

    cleaned_data.save_to_disk('../datasets/SA/icelandic/ice_and_fire_SA')

