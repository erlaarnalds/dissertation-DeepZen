from openai import OpenAI

from general import parse_label, sign


def rank_utterance(client, model_name, system_msg, utterance):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
          {"role": "system", "content": system_msg},
          {"role": "user", "content": utterance}
        ]
    )

    return response.choices[0].message.content



def eval(dataset, model_name):
    client = OpenAI()
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """

    labels = []
    predictions = []

    for utterance, label in dataset:

        pred = rank_utterance(client, model_name, system_msg, utterance)

        predictions.append(pred)

        true_val = parse_label(label, dataset.classes)
        labels.append(sign(true_val))
    
    return labels, predictions