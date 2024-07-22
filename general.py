
def sign(pred):
    threshold = 0.5

    if pred > threshold:
        return "positive"
    if pred < (threshold-1):
        return "negative"
    else:
        return "neutral"


def parse_label(label, classes):

    label_arr = label.split(";")

    scores = []

    for label in label_arr:
        scores.append(classes[label])
    
    return sum(scores) / len(label_arr)