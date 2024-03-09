import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import random

import main
from main import IMDB_Dataset


def test():
    # 0. Initialize model
    print("\nUsing only reviews with 50 or less words ")
    torch.manual_seed(1)
    np.random.seed(1)

    print("\nInitializing model ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification

    # Load the saved state_dict into the model
    model_load_path = './imdb_model/trained_model_state_dict.pth'
    try:
        model.load_state_dict(torch.load(model_load_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"Error: File not found at '{model_load_path}'. Please check the path and try again.")
        return
    
    print("Loaded model parameters from disk.")

    # 1. load data 
    print("\nLoading preprocessed train and test data ")
    test_file = "./data/imdb_test.csv"
    test_ds = IMDB_Dataset(test_file, tokenizer, subset_size = 200);

    bat_size = 20
    # train_ldr = DataLoader(train_ds, batch_size=bat_size, shuffle=True, drop_last=True, num_workers=1)
    test_ldr = DataLoader(test_ds, batch_size=bat_size, shuffle=False, drop_last=False)

    n_test = len(test_ds)
    print("Num test = %d " % (n_test))

    #READY to train model
    model.eval() 
    print("\nStarting inference on test data...")

    # Store predictions and actual labels for evaluation
    predictions = []
    actuals = []

    n_batch = 1

    # No gradient is needed for inference
    with torch.no_grad():
        for batch in test_ldr:

            print("\nLoading batch number %d" % (n_batch))
            n_batch += 1

            # Retrieve input data and labels, and move them to the CPU (if not already)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Perform a forward pass (predict with the model)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # The output logits are in `outputs.logits`
            # Use softmax to get probabilities, and argmax to get the predicted class
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions.extend(probs.argmax(dim=1).tolist())
            actuals.extend(labels.tolist())

    accuracy = sum(1 for x, y in zip(predictions, actuals) if x == y) / len(predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

test()