'''
This is the improved version of main.py
The main improvements are:
1. Now the input is a customizable csv, instead of hard coded in the text
2. Build a customizable training function.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


from pytorch_lstm_02minibatch.preprocess import seq_to_embedding, seqs_to_dictionary
from pytorch_lstm_02minibatch.model_lstm_tagger import LSTMTagger
from pytorch_lstm_02minibatch.train import train, test, plot_loss, train_val_test_split
from pytorch_lstm_02minibatch.preprocess import pad_sequences

torch.manual_seed(1)

# Usually 32 or 64 dim. Keeping them small
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Read in raw data
training_data_raw = pd.read_csv("./train.csv")

# Keep everything as np before training.
# Conversion to pytorch tensor will happen inside training.
texts = [t.split() for t in training_data_raw["text"].tolist()]
tags_list = [t.split() for t in training_data_raw["tag"].tolist()]
training_data = list(zip(texts, tags_list))

word_to_ix, tag_to_ix = seqs_to_dictionary(training_data)
print(tag_to_ix)

X_lens = np.array([len(x) for x in texts])
y_lens = np.array([len(y) for y in tags_list])

# Pad sequence to the desired length
X = pad_sequences([ seq_to_embedding(x, word_to_ix) for x in texts ], maxlen = 5,
                  padding = "post", value = 0)
y = pad_sequences([ seq_to_embedding(x, tag_to_ix) for x in tags_list], maxlen = 5,
                  padding = "post", value = 0)

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)-1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')


# Training
train_dataset, val_dataset, test_dataset = train_val_test_split(X, X_lens, y)
train_loss, val_loss = train(model, train_dataset, val_dataset, loss_fn, optimizer, n_epochs = 100)

# Examine training results
plot_loss(train_loss, val_loss)

# Calculate probability for test_dataset
tag_prob = test(model, test_dataset.tensors[0], test_dataset.tensors[2])

# convert to probability
print(tag_prob)
# Translate probability into tag index:
print(torch.argmax(tag_prob, dim=1))

