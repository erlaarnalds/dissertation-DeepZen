from datasets import Dataset
import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, dataset_path, dataset, shot, dataset_name=""):
        self.name = dataset_name
        self.index = 0
        self.path = dataset_path
        self.shot = shot
        self.train_ids = pd.read_csv(self.path + "/train_ids.csv").values.tolist()

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = pd.read_csv(dataset_path + "/test.csv")
    
    def parse_label(self, label, classes):

        label = label.lower()
        label_arr = label.split(";")

        scores = []

        for label in label_arr:
            scores.append(classes[label])
        
        return sum(scores) / len(label_arr)





class IceandFire(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):

        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            sentence = f"Snippet: {line['utterance']}"
            res = (line['id'], sentence, line['label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def __len__(self):
        return len(self.dataset)

    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Snippet: {row['utterance']}\n"
            example_str += f"{row['label']}\n\n"
        
        return example_str



class IceandFire_Irony(IceandFire):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_path=dataset_path, 
                         dataset_name=dataset_name, 
                         dataset=dataset,
                         shot=shot)

        self.classes = {'ekki kaldhæðin': -1.0, 'kaldhæðin': 1.0, 'óljós': 0.0}
        self.labels = ['irony', 'not irony', 'unclear']
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Irony Detection.
        You are given a sentence, and will evaluate whether it includes irony or not. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return IceandFire_Irony(dataset=dataset)
    
    
    def get_label(self, pred):

        threshold = 0.5

        if pred > threshold:
            return "irony"
        if pred < (threshold-1):
            return "not irony"
        else:
            return "unclear"





class IceandFire_ER(IceandFire):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_path=dataset_path, 
                         dataset_name=dataset_name, 
                         dataset=dataset,
                         shot=shot)


        self.classes = {'undrun': 0, 'fyrirlitning': 1, 'reiði': 2, 'sorg': 3, 'gleði': 4, 
            'hneykslun': 5, 'ógeðistilfinning': 6, 'hræðsla': 7, 'hlutlaust': 8}
        self.labels = ['surprise', 'contempt', 'anger', 'sadness', 'happiness', 'indignation', 'disgust', 'fear', 'neutral']
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Emotion Recognition.
        You are given a sentence, and will evaluate what emotion the sentence is portraying. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def filter_by_label(self, label):
        dataset = self.dataset[label == self.get_label(self.dataset['label'])].reset_index(drop=True)
        return IceandFire_ER(dataset=dataset)
    
    
    def get_label(self, pred):

        pred = np.asarray(pred)
        if np.all(pred ==  max(pred)): #no emotion ranked higher than others
            return "neutral"

        emotion_class = np.argmax(pred)

        return self.labels[emotion_class]
    
    def parse_label(self, label, classes):
        label = label.lower()
        label_arr = label.split(";")

        scores = np.zeros(len(classes))

        for label in label_arr:
            scores[classes[label]] += 1
        
        return scores / sum(scores)



class IceandFire_SA(IceandFire):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_path=dataset_path, 
                         dataset_name=dataset_name, 
                         dataset=dataset,
                         shot=shot)

        self.classes = {'neikvætt': -1.0, 'jákvætt': 1.0, 'hlutlaust': 0.0}
        self.labels = ['positive', 'negative', 'neutral']
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentiment analyser, and are tasked with reviewing sentences and analysing their overall sentiment.
        You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return IceandFire_SA(dataset=dataset)
    
    
    def get_label(self, pred):

        threshold = 0.5

        if pred > threshold:
            return "positive"
        if pred < (threshold-1):
            return "negative"
        else:
            return "neutral"


class IceandFire_hate(IceandFire):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_path=dataset_path, 
                         dataset_name=dataset_name, 
                         dataset=dataset,
                         shot=shot)

        self.classes = {'ekki hatursorðræða': -1.0, 'hatursorðræða': 1.0}
        self.labels = ['hate', 'not hate']
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Hate Detection.
        You are given a sentence, and will evaluate whether it includes hate speech or not. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return IceandFire_hate(dataset=dataset)
    
    
    def get_label(self, pred):

        threshold = 0

        if pred > threshold:
            return "hate"
        else:
            return "not hate"


class IceandFire_offense(IceandFire):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_path=dataset_path, 
                         dataset_name=dataset_name, 
                         dataset=dataset,
                         shot=shot)

        self.classes = {'ekki særandi': -1.0, 'særandi': 1.0}
        self.labels = ['offensive', 'not offensive']
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Offensive Detection.
        You are given a sentence, and will evaluate whether it includes offensive speech or not. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return IceandFire_offense(dataset=dataset)
    
    
    def get_label(self, pred):

        threshold = 0

        if pred > threshold:
            return "offensive"
        else:
            return "not offensive"


class RU(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'positive': 1.0}
        self.labels = ['positive', 'negative']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            sentence = f"Snippet: {line['utterance']}"
            res = (line['id'], sentence, line['label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return RU(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        
        base = f"""
        You are a highly capable sentiment analyser, and are tasked with reviewing documents and analysing their overall sentiment.
        You are given a sentence, and will output the class that you think captures the sentiment of the sentence. 
        The possible classes are: {classes}. 
        Do not output anything other than one of the classes listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples

    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Snippet: {row['utterance']}\n"
            example_str += f"{self.get_label(self.classes[row['label']])}\n\n"
        
        return example_str
    
    def get_label(self, pred):

        threshold = 0

        if pred > threshold:
            return "positive"
        else:
            return "negative"
    
    def __len__(self):
        return len(self.dataset)



class REST14(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}
        self.labels = ['positive', 'negative', 'neutral']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            sentence = f"Snippet: {line['utterance']}\nAspect: {line['aspect']}"
            res = (line['original_id'], sentence, line['aspect_label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['aspect_label'] == label].reset_index(drop=True)
        return REST14(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        
        base = f"""
        You are a highly capable sentiment analyser, please perform Aspect Sentiment Classification.
        You are given a review for a restaurant, and a specific aspect to evaluate. You will output the class that you think captures the sentiment of the aspect in the review. 
        The possible classes are: {classes}.
        Do not output anything other than one of the classes listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Snippet: {row['utterance']}\n"
            example_str += f"Aspect: {row['aspect']}\n"
            example_str += f"{self.get_label(self.classes[row['label']])}\n\n"
        
        return example_str
    
    def get_label(self, pred):

        threshold = 0.5

        if pred > threshold:
            return "positive"
        if pred < (threshold-1):
            return "negative"
        else:
            return "neutral"
    
    def __len__(self):
        return len(self.dataset)


class CompSent19(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'worse': -1.0, 'better': 1.0}
        self.labels = ['worse', 'better']

    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]

            tuple_str = line['tuple']
            tuple_arr = [i[1:-1] for i in tuple_str[1:-1].split(", ")]

            sentence = f"Compare {tuple_arr[0]} to {tuple_arr[1]}. Snippet: {line['utterance']}"
            res = (line['original_id'], sentence, tuple_arr[2].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['aspect_label'] == label].reset_index(drop=True)
        return CompSent19(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        
        base = f"""
        You are a highly capable sentiment analyser, please perform Comparative Opinion evaluation.
        You will be given a sentence, as well as two objects to compare. Assign the sentence one of either class: {classes}.
        Do not output anything other than one of the classes listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            tuple_str = row['tuple']
            tuple_arr = [i[1:-1] for i in tuple_str[1:-1].split(", ")]

            example_str += f"Compare {tuple_arr[0]} to {tuple_arr[1]}. Snippet: {row['utterance']}\n"
            example_str += f"{self.get_label(self.classes[row['label']])}\n\n"
        
        return example_str
    
    def get_label(self, pred):

        threshold = 0.0

        if pred > threshold:
            return "better"
        else:
            return "worse"
    
    def __len__(self):
        return len(self.dataset)


class REST16_UABSA(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'neutral':0.0, 'positive': 1.0}
        self.labels = ['negative', 'neutral', 'positive']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]

            sentence = f"Review: {line['utterance']}"
            res = (line['original_id'], sentence, line['label'])
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['aspect_label'] == label].reset_index(drop=True)
        return REST16_UABSA(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        
        base = f"""
        You are a highly capable sentiment analyser, please perform Unified Aspect-Based Sentiment Analysis.
        You will be given a restaurant review. Please tag all (aspect, sentiment) pairs and order them in a list.
        The aspect should occur in the review, and the sentiment should be one of the following classes: {classes}.
        If there are no aspect-sentiment pairs, return an empty list. Otherwise, return a python list of the sentiment pairs as tuples.
        Do not output anything other than the array of (aspect, sentiment) pairs. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Review: {row['utterance']}\n"
            example_str += f"{row['label']}\n\n"
        
        return example_str
    
    
    def __len__(self):
        return len(self.dataset)


class ASTE(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'neutral':0.0, 'positive': 1.0}
        self.labels = ['negative', 'neutral', 'positive']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]

            sentence = f"Review: {line['utterance']}"
            res = (line['original_id'], sentence, line['label'])
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['aspect_label'] == label].reset_index(drop=True)
        return ASTE(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        
        base = f"""
        You are a highly capable sentiment analyser, please perform Aspect Sentiment Triplet Extraction.
        You will be given a restaurant review. Please tag all (aspect, opinion, sentiment) triplets and order them in a list.
        The aspect and opinion should occur in the review, and the sentiment should be one of the following classes: {classes}.
        Return a python list of the sentiment triplets as tuples, with aspect, opinion and sentiment all enclosed in single quotes.
        Do not output anything other than the array of (aspect, opinion, sentiment) triplets. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Review: {row['utterance']}\n"
            example_str += f"{row['label']}\n\n"
        
        return example_str
    
    
    def __len__(self):
        return len(self.dataset)


class ASQP(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'neutral':0.0, 'positive': 1.0}
        self.labels = ['negative', 'neutral', 'positive']
        self.categories =  ['andrúmsloft almennt', 'drykkir verð',
            'drykkir gæði', 'drykkir úrval', 'matur almennt', 'matur verð', 'matur gæði',
            'matur úrval', 'staðsetning almennt', 'veitingastaður almennt', 'veitingastaður alls_konar',
            'veitingastaður verð', 'þjónusta almennt']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]

            sentence = f"Review: {line['utterance']}"
            res = (line['original_id'], sentence, line['label'])
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['aspect_label'] == label].reset_index(drop=True)
        return REST14(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
        categories_str = ", ".join(self.categories)
        
        base = f"""
        You are a highly capable sentiment analyser, please perform Aspect Sentiment Quadruplet Prediction.
        You will be given a restaurant review. Please tag all (category, aspect, opinion, sentiment) quadruplets and order them in a list.
        The category should be one of the following: {categories_str}. The aspect and opinion should occur in the review and the sentiment should be one of the following classes: {classes}.
        Only aspect can be 'NULL' if no appropriate aspect can be found, category, opinion and sentiment cannot be 'NULL'. 
        Return a python list of the sentiment quadruplets as tuples, with category, aspect, opinion and sentiment all enclosed in single quotes.
        Do not output anything other than the array of (category, aspect, opinion, sentiment) quadruplets. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Review: {row['utterance']}\n"
            example_str += f"{row['label']}\n\n"
        
        return example_str
    
    
    def __len__(self):
        return len(self.dataset)



class Stance(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'against': -1.0, 'favor': 1.0, 'none': 0.0}
        self.labels = ['against', 'favor', 'none']
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            sentence = f"Evaluate stance regarding {line['domain']}. Snippet: {line['utterance']}"
            res = (line['original_id'], sentence, line['label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return Stance(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Stance Detection.
        You are given a sentence, and will evaluate what stance it has towards a certain aspect. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Evaluate stance regarding {row['domain']}. Snippet: {row['utterance']}\n"
            example_str += f"{self.get_label(self.classes[row['label'].lower()])}\n\n"
        
        return example_str
    
    def get_label(self, pred):

        threshold = 0.5

        if pred > threshold:
            return "favor"
        if pred < (threshold-1):
            return "against"
        else:
            return "none"

    
    def __len__(self):
        return len(self.dataset)


class Implicit(Dataset):
    def __init__(self, dataset_path="", dataset_name="", dataset=None, shot=0):
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, dataset=dataset, shot=shot)

        self.classes = {'negative': -1.0, 'positive': 1.0, 'neutral': 0.0}
        self.labels = ['negative', 'positive', 'neutral']

    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            sentence = f"Infer sentiment regarding the aspect: {line['aspect']}. Snippet: {line['utterance']}"
            res = (line['original_id'], sentence, line['label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
    def __getitem__(self, ind):
        return self.dataset.iloc[ind]['utterance']
    
    def filter_by_label(self, label):
        dataset = self.dataset[self.dataset['label'] == label].reset_index(drop=True)
        return Implicit(dataset=dataset)
        
    
    def get_system_msg(self):
        classes = ", ".join(self.labels)
                
        base =  f"""
        You are a highly capable sentence analyser, please perform Aspect-Based Implicit Sentiment Analysis.
        You are given a sentence, and will evaluate what the implicit sentimenti is regarding a certain aspect. 
        The possible classes are: {classes}. 
        Do not output any other class than the ones listed. DO NOT explain the reasoning for your classification.
        """

        if self.shot == 0:
            return base
        
        else:
            examples = self.get_few_shot_examples()
            return base + "\nExamples:\n"  + examples


    def get_few_shot_examples(self):
        example_str = ""
        examples = pd.read_csv(self.path + f"/few_shot_{self.shot}.csv")

        for ind, row in examples.iterrows():
            example_str += f"Infer sentiment regarding the aspect: {row['aspect']}. Snippet: {row['utterance']}"
            example_str += f"{self.get_label(self.classes[row['label'].lower()])}\n\n"
        
        return example_str
    
    def get_label(self, pred):

        threshold = 0.5

        if pred > threshold:
            return "positive"
        if pred < (threshold-1):
            return "negative"
        else:
            return "neutral"

    
    def __len__(self):
        return len(self.dataset)