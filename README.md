# ML_Imdb_Review

An experiment with machine learning using:
- Pretrained Bert model
- IMDB movie review dataset from torchtext

Credits to:
- "IMDB Classification using PyTorch Transformer Architecture", James D. McCaffery, March 16 2022

To test out the code:
1. run get_imdb_dataset.py to extract IMDB movie review in the form of csv
2. run main.py to train a model based on Bert  
  Note: the model runs on cpu, switch to gpu if you have one
3. run ml_test.py to test accuracy