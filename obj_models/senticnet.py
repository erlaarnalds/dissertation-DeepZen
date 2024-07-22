import re
from general import parse_label, sign


def preprocess(sentence):
    # switch out icelandic chars as they are not present in senticnet
    ice_to_eng = {
    'á': 'a', 'ð': 'd', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ý': 'y', 'þ': 'th', 'æ': 'ae', 'ö': 'o',
    'Á': 'A', 'Ð': 'D', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ý': 'Y', 'Þ': 'Th', 'Æ': 'Ae', 'Ö': 'O'
    }
    for ice_char, eng_char in ice_to_eng.items():
        sentence = sentence.replace(ice_char, eng_char) 

    # remove anything that is not a word character or white space
    words = re.sub(r'[^\w\s]', '', sentence).lower().split()

    return words

def eval(senticnet, dataset):

    labels = []
    predictions = []
    
    for utterance, label in dataset:
        sentiment_scores = []
        words = preprocess(utterance)

        for word in words:
            if word in senticnet:
                # get polarity value of word
                sentiment_scores.append(senticnet[word][7])
    
        if len(sentiment_scores) > 0:
            score = sum(sentiment_scores) / len(sentiment_scores)
        else:
            score = 0.0
        predictions.append(sign(score))
        true_val = parse_label(label, dataset.classes)
        labels.append(sign(true_val))

    return labels, predictions