# TTS evaluations

This folder includes code to successfully generate audio for TTS comparison tests. First, install the necessary package requirements:

```
pip install -r requirements.txt 
````

To create the audio, access to the dataset Talrómur is required. You can install it here: https://repository.clarin.is/repository/xmlui/handle/20.500.12537/104

Unzip the alfur, dilja and bui files and add to the relevant folders under *data/*

Generate examples by running:
```
python test_example_generation.py
````

This will create audio samples in *voice_samples/* along with an index.tsv file with metadata, suitable for uploading to the MOSI platform for MOS testing (see https://github.com/cadia-lvl/MOSI).

## SECS score

To evaluate SECS score, run

```
python secs_calucalation.py
````

This code will create comparison samples and then evaluate their SECS score. 


## Fine tuning models using Talrómur and MMS

Start by creating a dataset on huggingface to fine-tune on using Talrómur, or other voice samples. The code in *create_datasets.py* will help with that if you wish to use Talrómur.

Open *create_datasets.py* and edit the voice and hf_user parameters.

Then run:

```
python create_datasets.py
````

Once that has finished, you can continue on to fine-tuning.

Clone the following repository: https://github.com/ylacombe/finetune-hf-vits/tree/main

Follow step 1 in the README provided in the repository. 

Next, edit the *finetune_isl.json* to specific fine-tuning parameters to your liking. Specifically edit the output HuggingFace file (hub_model_id) to a name for your liking, and change dataset_name to the dataset you created. The file also assumes that 2000 voice samples will be used. To change this, change the max_train_samples parameter.


Finally, start fine-tuning by running the following command:

```
accelerate launch <path_to_finetuning_repo>/run_vits_finetuning.py ./finetune_isl.json
````











