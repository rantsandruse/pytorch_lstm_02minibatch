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

def train_val_test_split(X, X_lens, y, train_val_split = 10, trainval_test_split = 10):
    '''
    Pre-split data to train, val and test.
    Parameters
    ----------
    X
    X_lens
    y
    train_val_split
    trainval_test_split

    Returns
    -------

    '''
    # We switch over to stratified kfold
    splits = KFold(n_splits=trainval_test_split, shuffle=True, random_state=42)
    for trainval_idx, test_idx in splits.split(X, y):
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        X_lens_trainval, X_lens_test = X_lens[trainval_idx], X_lens[test_idx]

    splits = KFold(n_splits=train_val_split, shuffle=True, random_state=28)

    for train_idx, val_idx in splits.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        X_lens_train, X_lens_val = X_lens_trainval[train_idx], X_lens_trainval[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype = torch.long),
                                  torch.tensor(y_train, dtype=torch.long),
                                  torch.tensor(X_lens_train, dtype=torch.int64))

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                torch.tensor(y_val, dtype=torch.long),
                                torch.tensor(X_lens_val, dtype=torch.int64))

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long),
                                 torch.tensor(y_test, dtype=torch.long),
                                 torch.tensor(X_lens_test, dtype=torch.int64))


    return train_dataset, val_dataset, test_dataset


def train(model, train_dataset, val_dataset, loss_fn, optimizer, n_epochs = 5, batch_size = 2,
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
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, drop_last=True)
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(n_epochs):
        train_losses = []
        val_losses = []
        for X_batch, y_batch, X_lens_batch in train_loader:
            optimizer.zero_grad()
            ypred_batch = model(X_batch, X_lens_batch)

            loss = loss_fn(ypred_batch.view(batch_size * seq_len, -1),
                           y_batch.view(batch_size * seq_len))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        with torch.no_grad():
            for X_val, y_val, X_lens_val in val_loader:
                ypred_val = model(X_val, X_lens_val)

                # flatten first
                val_loss = loss_fn(ypred_val.view(batch_size*seq_len, -1), y_val.view(batch_size * seq_len))
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
