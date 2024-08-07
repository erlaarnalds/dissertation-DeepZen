from general import parse_label, parse_label_ER
from anthropic import Anthropic
import os
import time

def rank_utterance(client, model_name, system_msg, utterance):

    rate_limit_per_minute = 50
    delay = 60.0 / rate_limit_per_minute

    # slow down requests to prevent rate limiting
    time.sleep(delay)

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
    #print("Tokens remaining: ",response.headers.get('anthropic-ratelimit-tokens-remaining'))
    
    return response.content[0].text.lower()


def eval(dataset, model_name, task):
    client = Anthropic()
    system_msg = dataset.get_system_msg()

    labels = []
    predictions = []

    for utterance, label in dataset:

        if not utterance.isspace() and utterance != "" and utterance != None:
            pred = rank_utterance(client, model_name, system_msg, utterance)
            predictions.append(pred)

            if task == "ABSA":
                true_label = label
            elif dataset.name == "ice_and_fire_ER":
                true_val = parse_label_ER(label, dataset.classes)
                true_label = dataset.get_label(true_val)
            else:
                true_val = parse_label(label, dataset.classes)
                true_label = dataset.get_label(true_val)
            
            labels.append(true_label)


            if not os.path.isfile(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv"):
                with open(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv", "w") as file:
                    file.write(f"utterance, prediction, true label\n")

            with open(f"logs/{dataset.name}/{model_name}_shot_{dataset.shot}.csv", "a") as file:
                file.write(f"{utterance}, {pred}, {true_label}\n")
    
    return labels, predictions
