import numpy as np


def parse_label(label, classes):

    label_arr = label.split(";")

    scores = []

    for label in label_arr:
        scores.append(classes[label])
    
    return sum(scores) / len(label_arr)

def parse_label_ER(label, classes):

    label_arr = label.split(";")

    scores = np.zeros(len(classes))

    for label in label_arr:
        scores[classes[label]] += 1
    
    return scores / sum(scores)


def get_system_msg(dataset):

    classes = ""
    for label, rating in dataset.classes.items():
        classes += label + ", "
    
    classes = classes[:-2]

    return f"""
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: {classes}. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """