'''
Training, testing and loss functions.
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


def train(model, loss_fn, X, y, X_lens, optimizer, n_epochs = 5, batch_size = 2,
          seq_len = 5):
    '''
    Parameters
    ----------
    model
    X
    y
    X_lens
    optimizer
    loss_fn
    n_epochs
    batch_size
    seq_len

    Returns
    epoch_train_losses: Training loss over epoches
    epoch_val_losses: Validation loss over epoches
    -------

    '''
    # Use scikit learn stratified k-fold.
    # I gave up on the initial choice of pytorch random_split, as it would not return indices.
    # train_dataset, val_dataset = random_split(dataset, [16,4])
    splits = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in splits.split(X,y):
        X_train, X_test = torch.tensor(X[train_idx], dtype = torch.long), \
                          torch.tensor(X[test_idx], dtype = torch.long)
        y_train, y_test = torch.tensor(y[train_idx], dtype = torch.long), \
                          torch.tensor(y[test_idx], dtype = torch.long)
        X_lens_train, X_lens_test = torch.tensor(X_lens[train_idx], dtype=torch.float), \
                                    torch.tensor(X_lens[test_idx], dtype = torch.float)

    train_dataset = TensorDataset(X_train, y_train, X_lens_train)
    test_dataset = TensorDataset(X_test, y_test, X_lens_test)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
    val_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(n_epochs):
        train_losses = []
        val_losses = []
        for X_batch, y_batch, X_lens_batch in train_loader:
            optimizer.zero_grad()
            ypred_batch = model(X_batch, X_lens_batch)

            # flatten y_batch and ypred_batch
            y_batch = y_batch.view(batch_size * seq_len)
            ypred_batch = ypred_batch.view(batch_size * seq_len, -1)

            loss = loss_fn(ypred_batch.view(batch_size*seq_len, -1),
                           y_batch.view(batch_size * seq_len))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        with torch.no_grad():
            for X_val, y_val, X_lens_val in val_loader:
                ypred_val = model(X_val, X_lens_val)

                # flatten first
                ypred_val = ypred_val.view(batch_size*seq_len, -1)
                y_val = y_val.view(batch_size * seq_len)

                val_loss = loss_fn(ypred_val,y_val)
                val_losses.append(val_loss.item())

        epoch_train_losses.append(np.mean(train_losses))
        epoch_val_losses.append(np.mean(val_losses))

    return epoch_train_losses, epoch_val_losses


def test(model, X_test, X_lengths_test):
    '''
    Inference function
    Parameters
    ----------
    model: LSTM tagger model
    X_test: Test data
    X_lengths_test: Original Test data

    Returns post-softmax probability of individual classes.
    -------

    '''
    with torch.no_grad():
        tag_scores = model(X_test, X_lengths_test)
        # Now evaluate probabilistic output
        # For either NLL loss or cross entropy los
        # Use cross entropy loss
        tag_prob = F.softmax(tag_scores, dim = 1)
        return tag_prob

def plot_loss(train_loss, val_loss):
    '''
    Visualize training loss vs. validation loss.
    Parameters
    ----------
    train_loss: training loss
    val_loss: validation loss

    Returns: None
    -------

    '''
    loss_csv = pd.DataFrame({"iter": range(len(train_loss)), "train_loss": train_loss,
                             "val_loss": val_loss})
    loss_csv.to_csv("./output/loss.csv")
    # gca stands for 'get current axis'
    ax = plt.gca()
    loss_csv.plot(kind='line',x='iter',y='train_loss',ax=ax )
    loss_csv.plot(kind='line',x='iter',y='val_loss', color='red', ax=ax)
    # plt.show()
    plt.savefig("./output/train_vs_val_loss.png")
