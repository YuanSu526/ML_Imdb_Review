import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import random


# -----------------------------------------------------------

class IMDB_Dataset(torch.utils.data.Dataset):
    def __init__(self, src_file, tokenizer, max_length=512, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_csv(src_file, usecols=['sentiment', 'review'])
        
        if subset_size is not None:
            df = df.sample(n=subset_size, random_state=42)  # For reproducibility
        
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx] - 1  # Adjust labels to 0 and 1
        }


# A placeholder for the actual tokenizer function
def simple_tokenizer(review, max_length):
    # This function should be replaced with an actual tokenizer that converts text to token IDs.
    # Here we'll just split the text into words for demonstration purposes.
    return review.split()[:max_length]
# -----------------------------------------------------------

def main():
    # 0. Initialize model
    print("\nBegin PyTorch IMDB Transformer Architecture demo ")
    print("\nUsing only reviews with 50 or less words ")
    torch.manual_seed(1)
    np.random.seed(1)

    print("\nInitializing model ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification
    device = torch.device("cpu")
    model.to(device)

    # 1. load data 
    print("\nLoading preprocessed train and test data ")
    train_file = "./data/imdb_train.csv"
    train_ds = IMDB_Dataset(train_file, tokenizer, subset_size = 1000); 

    bat_size = 20
    # train_ldr = DataLoader(train_ds, batch_size=bat_size, shuffle=True, drop_last=True, num_workers=1)
    train_ldr = DataLoader(train_ds, batch_size=bat_size, shuffle=True, drop_last=True)

    n_train = len(train_ds)
    print("Num train = %d Num test = %d " % (n_train))

    #READY to train model
    print("\nStart training model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()

    n_batch = 1

    for epoch in range(2):  # Example: loop over the dataset 2 times
        total_loss = 0
        for step, batch in enumerate(train_ldr):
            print("\nLoading batch number %d" % (n_batch))
            n_batch += 1

            # Reset gradients
            model.zero_grad()

            # Retrieve input data; adjust labels
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass to calculate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_ldr)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model_save_path = './imdb_model/trained_model_state_dict.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

    #TODO: implement testing loop or keep training, also research on how to use stored parameters


    print("\nEnd")

if __name__ == "__main__":
    main()