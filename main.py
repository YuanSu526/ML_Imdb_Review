import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# -----------------------------------------------------------

class IMDB_Dataset(torch.utils.data.Dataset):
    def __init__(self, src_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_csv(src_file, usecols=['sentiment', 'review'])
        
        self.labels = torch.tensor(df['sentiment'].values, dtype=torch.long)
        self.reviews = df['review'].values
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.reviews[idx],
                                  return_tensors='pt',
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length)
        
        return {
            #The encoded version of reviews
            'input_ids': encoding['input_ids'].squeeze(0),  

            #Sequence of 1 and 0 indicating which token in input_ids 
            #should be looked at or ignored
            'attention_mask': encoding['attention_mask'].squeeze(0), 

            'label': self.labels[idx]
        }


# A placeholder for the actual tokenizer function
def simple_tokenizer(review, max_length):
    # This function should be replaced with an actual tokenizer that converts text to token IDs.
    # Here we'll just split the text into words for demonstration purposes.
    return review.split()[:max_length]
# -----------------------------------------------------------

def main():
    # 0. get started
    print("\nBegin PyTorch IMDB Transformer Architecture demo ")
    print("Using only reviews with 50 or less words ")
    torch.manual_seed(1)
    np.random.seed(1)

    #Initialize model
    print("\nInitializing model ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification
    device = torch.device("cpu")
    model.to(device)

    # 1. load data 
    print("\nLoading preprocessed train and test data ")
    train_file = "./data/imdb_train.csv"
    train_ds = IMDB_Dataset(train_file, tokenizer) 
    test_file = "./data/imdb_test.csv"
    test_ds = IMDB_Dataset(test_file, tokenizer)
    
    # print("\nDisplaying first 3 samples from training data:")
    # for i in range(3):
    #     sample = train_ds[i]
    #     print(f"Sample {i}:")
    #     print(f"Label: {'Positive' if sample['label'] == 1 else 'Negative'}")
    #     print(f"InputId: {sample['input_ids']}")
    #     print()

    bat_size = 20
    train_ldr = DataLoader(train_ds, batch_size=bat_size, shuffle=True, drop_last=True)
    test_ldr = DataLoader(test_ds, batch_size=bat_size, shuffle=False, drop_last=False)

    n_train = len(train_ds)
    n_test = len(test_ds)
    print("Num train = %d Num test = %d " % (n_train, n_test))


    #READY to train model

    print("\nEnd")

if __name__ == "__main__":
    main()