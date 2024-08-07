import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from general import parse_label, parse_label_ER


def rank_utterance(client, model_name, system_msg, utterance):

    input_text = "[INST]" + system_msg + "[/INST]\n" + utterance

    chat_response = client.chat(
        model=model_name,
        messages=[ChatMessage(role="user", content=input_text)]
    )

    return chat_response.choices[0].message.content.lower()



def eval(dataset, model_name, task=""):
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    system_msg = dataset.get_system_msg()

    labels = []
    predictions = []

    for utterance, label in dataset:

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