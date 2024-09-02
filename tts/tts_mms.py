from transformers import VitsModel, VitsTokenizer
import torch
import scipy
import numpy as np

class MMSClient():
    def __init__(self, model, model_name, model_id):
        self.model = VitsModel.from_pretrained(model)
        self.tokenizer = VitsTokenizer.from_pretrained(model)
        self.model_name = model_name
        self.model_id = model_id


    def generate(self, text, output_dir):

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Convert the output tensor to a numpy array
        waveform_np = output.squeeze().cpu().numpy()

        # Ensure the waveform is in the correct format (float32 and normalized)
        waveform_np = waveform_np.astype(np.float32)


        scipy.io.wavfile.write(output_dir, rate=self.model.config.sampling_rate, data=waveform_np)

