from datasets import Dataset
import pandas as pd

class IceandFire():
    def __init__(self, dataset_path, dataset_name):
        self.dataset = Dataset.load_from_disk(dataset_path)
        self.index = 0
        self.name = dataset_name
        self.classes = {'neikvætt': -1.0, 'jákvætt': 1.0, 'hlutlaust': 0.0}
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset[self.index]
            res = (line['comment_body'], line['label'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
    
        
class IMDB():
    def __init__(self, dataset_path, dataset_name):
        self.dataset = pd.read_csv(dataset_path + ".csv")
        # select only 500
        self.dataset = self.dataset.head(500)
        self.index = 0
        self.name = dataset_name
        self.classes = {'negative': -1.0, 'positive': 1.0, 'neutral': 0.0}
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            line = self.dataset.iloc[self.index]
            res = (line['review'], line['sentiment'].lower())
            self.index += 1
            return res
        else:
            raise StopIteration
