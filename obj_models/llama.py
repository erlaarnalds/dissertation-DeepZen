import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def eval(dataset, model_name):
    # Load the tokenizer


    

    # Decode the generated text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Generated Text:")
    print(output_text)

def eval(dataset, model_name):
    system_msg = """
    You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
    You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
    The possible classes are: Positive, Negative, Neutral. 
    Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
    """

    labels = []
    predictions = []

    model_id = "meta-llama/" + model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for utterance, label in dataset:
        print(utterance)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": utterance},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))

    #     # Generate text using the model
    #     pred = model.generate(input_ids, max_length=50, num_return_sequences=1)
    #     print(pred)
    #     predictions.append(pred)

    #     true_val = parse_label(line['label'])
    #     labels.append(sign(true_val))
    
    # return labels, predictions