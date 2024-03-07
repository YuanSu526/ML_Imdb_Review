# https://jamesmccaffrey.wordpress.com/2022/03/16/imdb-classification-using-pytorch-transformer-architecture/
# imdb_transformer.py
# Transformer Architecture classification for IMDB
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11

import numpy as np
import torch as T
import math

device = T.device('cpu')

# -----------------------------------------------------------

class TransArch_Net(T.nn.Module):
  def __init__(self):
    # vocab_size = 129892
    super(TransArch_Net, self).__init__()
    self.embed = T.nn.Embedding(129892, 32)  # word embedding
    self.pos_enc = \
      PositionalEncoding(32, dropout=0.00)  # positional
    self.enc_layer = T.nn.TransformerEncoderLayer(d_model=32,
      nhead=2, dim_feedforward=100, 
      batch_first=True)  # d_model divisible by nhead
    self.trans_enc = T.nn.TransformerEncoder(self.enc_layer,
      num_layers=6)
    self.fc1 = T.nn.Linear(32*50, 2)  # 0=neg, 1=pos
 
  def forward(self, x):
    # x = review/sentence. length = fixed w/ padding
    z = self.embed(x)  # tokens to embed vector
    z = z.reshape(-1, 50, 32)  # bat seq embed 
    # z = self.pos_enc(z) * np.sqrt(32)  # hurts
    z = self.pos_enc(z) 
    z = self.trans_enc(z) 
    z = z.reshape(-1, 32*50)  # torch.Size([bs, 1600])
    z = T.log_softmax(self.fc1(z), dim=1)  # NLLLoss()
    return z 

# -----------------------------------------------------------

class PositionalEncoding(T.nn.Module):  # documentation code
  def __init__(self, d_model: int, dropout: float=0.1,
   max_len: int=5000):
    super(PositionalEncoding, self).__init__()  # old syntax
    self.dropout = T.nn.Dropout(p=dropout)
    pe = T.zeros(max_len, d_model)  # like 10x4
    position = \
      T.arange(0, max_len, dtype=T.float).unsqueeze(1)
    div_term = T.exp(T.arange(0, d_model, 2).float() * \
      (-np.log(10_000.0) / d_model))
    pe[:, 0::2] = T.sin(position * div_term)
    pe[:, 1::2] = T.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)  # allows state-save

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

# -----------------------------------------------------------

class IMDB_Dataset(T.utils.data.Dataset):
  # 50 token IDs then 0 or 1 label, space delimited
  def __init__(self, src_file):
    # super().__init__(IMDB_Dataset)  # not needed
    all_xy = np.loadtxt(src_file, usecols=range(0,51),
      delimiter=" ", comments="#", dtype=np.int64)
    tmp_x = all_xy[:,0:50]   # cols [0,50) = [0,49]
    tmp_y = all_xy[:,50]     # all rows, just col 50
    self.x_data = T.tensor(tmp_x, dtype=T.int64) 
    self.y_data = T.tensor(tmp_y, dtype=T.int64) 

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    tokens = self.x_data[idx]
    trgts = self.y_data[idx] 
    return (tokens, trgts)

# -----------------------------------------------------------

def accuracy(model, dataset):
  # assumes model.eval()
  n_correct = 0; n_wrong = 0
  ldr = T.utils.data.DataLoader(dataset,
    batch_size=1, shuffle=False)
  for (batch_idx, batch) in enumerate(ldr):
    X = batch[0]  # inputs
    Y = batch[1]  # target sentiment label
    with T.no_grad():
      oupt = model(X)  # log-probs
   
    idx = T.argmax(oupt.data)
    if idx == Y:  # predicted == target
      n_correct += 1
    else:
      n_wrong += 1
  acc = (n_correct * 100.0) / (n_correct + n_wrong)
  return acc

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin PyTorch IMDB Transformer Architecture demo ")
  print("Using only reviews with 50 or less words ")
  T.manual_seed(1)
  np.random.seed(1)

  # 1. load data 
  print("\nLoading preprocessed train and test data ")
  train_file = ".\\Data\\imdb_train_50w.txt"
  train_ds = IMDB_Dataset(train_file) 

  test_file = ".\\Data\\imdb_test_50w.txt"
  test_ds = IMDB_Dataset(test_file) 

  bat_size = 20
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True, drop_last=True)
  n_train = len(train_ds)
  n_test = len(test_ds)
  print("Num train = %d Num test = %d " % (n_train, n_test))

# -----------------------------------------------------------

  # 2. create network
  net = TransArch_Net().to(device)

  # 3. train model
  loss_func = T.nn.NLLLoss()  # log-softmax() activation
  optimizer = T.optim.Adam(net.parameters(), lr=1.0e-3)
  max_epochs = 200
  log_interval = 20  # display progress 

  print("\nbatch size = " + str(bat_size))
  print("loss func = " + str(loss_func))
  print("optimizer = Adam ")
  print("learn rate = 0.001 ")
  print("max_epochs = %d " % max_epochs)

  print("\nStarting training ")
  net.train()  # set training mode
  for epoch in range(0, max_epochs):
    T.manual_seed(1 + epoch)  # for reproducibility
    tot_err = 0.0  # for one epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch[0]
      Y = batch[1]
      optimizer.zero_grad()
      oupt = net(X)

      loss_val = loss_func(oupt, Y) 
      tot_err += loss_val.item()
      loss_val.backward()  # compute gradients
      optimizer.step()     # update weights
  
    if epoch % log_interval == 0:
      print("epoch = %4d  |" % epoch, end="")
      print("   loss = %12.6f  |" % tot_err, end="")
      net.eval()
      train_acc = accuracy(net, train_ds)
      print("  accuracy = %8.2f%%" % train_acc)
      net.train()

      # net.eval()
      # test_acc = accuracy(net, test_ds)
      # print("  accuracy = %8.2f%% \n" % test_acc)
      # net.train()

  print("Training complete")

# -----------------------------------------------------------

  # 4. evaluate model
  net.eval()
  test_acc = accuracy(net, test_ds)
  print("\nAccuracy on test data = %8.2f%%" % test_acc)

  # 5. save model
  print("\nSaving trained model state")
  fn = ".\\Models\\imdb_model.pt"
  T.save(net.state_dict(), fn)

  # saved_model = Net()
  # saved_model.load_state_dict(T.load(fn))
  # use saved_model to make prediction(s)

  # 6. use model, hard-coded approach
  print("\nSentiment for \"the movie was a great \
waste of my time\"")
  print("0 = negative, 1 = positive ")
  review_ids = [4, 20, 16, 6, 86, 425, 7, 58, 64]
  padding = np.zeros(50-len(review_ids), dtype=np.int64)
  review = np.concatenate([padding, review_ids])
  review = T.tensor(review, dtype=T.int64).to(device)
  
  net.eval()
  with T.no_grad():
    prediction = net(review)  # log-probs
  print("raw output : ", end=""); print(prediction)
  print("pseud-probs: ", end=""); print(T.exp(prediction))

  # 6. use model, programmatic approach 
  # assumes you have a file of word-rank vocabulary info
 
  # print("\nSentiment for \"the movie was a great \
  # waste of my time\"")
  # print("0 = negative, 1 = positive ")
  # word_to_id = {}   # word to ID
  # # get saved vocabulary data - word space 1-based rank
  # f = open(".\\Data\\vocab_file.txt", 'r', encoding='utf8')
  # for line in f:
  #   line = line.strip()  # remove trailing newline
  #   tokens = line.split(" ")
  #   w = tokens[0]
  #   rank = int(tokens[1])  # "the" = 1, etc.
  #   word_to_id[w] = rank + 3  # "the" = 4, etc.
  # f.close()
  # word_to_id[0] = "(PAD)"
  # word_to_id[1] = "(SOS)"
  # word_to_id[2] = "(OOV)"
  # word_to_id[3] is not used

  # review = "the movie was a great waste of my time"
  # review_ids = []
  # wrds = review.split(" ")
  # for w in wrds:
  #   id = word_to_id[w]
  #   review_ids.append(id)
  # # review_ids is [4, 20, 16, 6, 86, 425, 7, 58, 64]

  # padding = np.zeros(41, dtype=np.int64)
  # review = np.concatenate([padding, review_ids])
  # review = T.tensor(review, dtype=T.int64).to(device)
  
  # net.eval()
  # with T.no_grad():
  #   prediction = net(review)  # log-probs
  # print("raw output : ", end=""); print(prediction)
  # print("pseud-probs: ", end=""); print(T.exp(prediction))

  print("\nEnd PyTorch IMDB Transformer demo")

if __name__ == "__main__":
  main()