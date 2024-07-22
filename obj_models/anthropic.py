from general import parse_label, sign
from anthropic import Anthropic

def rank_utterance(client, model_name, system_msg, utterance):

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        system=system_msg,
        messages=[
          {"role": "user", 
           "content": [
               {
                   "type": "text",
                   "text": utterance
               }
           ]}
        ]
    )

    return response.content[0].text


def eval(dataset, model_name):
    client = Anthropic()
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """

    labels = []
    predictions = []

    for utterance, label in dataset:

        if not utterance.isspace() and utterance != "" and utterance != None:
            pred = rank_utterance(client, model_name, system_msg, utterance)
            predictions.append(pred)

            true_val = parse_label(label, dataset.classes)
            labels.append(sign(true_val))
    
    return labels, predictions
