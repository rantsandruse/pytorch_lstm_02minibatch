# Learning Pytorch in Ten Days: Day 2 - Train an LSTM model in minibatch (with proper initialization and padding) 

In day 1 tutorial, we've learned how to work with a very simple LSTM network, by training the model on all available data over 
multiple epochs. In this tutorial, I will show you how to train an LSTM model in minibatches, with proper 
variable initialization and padding. Again, I will break it down into Q & A format: 

 ## Why do we need to train in mini-batches?   
Deep neural networks are data hungry and need to be trained with a large volume of data. 
If you were to train your deep neural network in a single batch, there are a couple of problems: 
1. All of your data may not fit in memory. You will likely run into an out-of-memory error from the get-go. 
2. Now assume that you have all the memory you desire and will never run into OOM error, training all data in a single batch 
   is still far from ideal. Prior research has shown that a large mini-batch based training routine will oftentimes lead 
   you to a sharp minima when you have a rugged loss function landscape via stochastic gradient descent, and become 
   irreversible stuck there. We will discuss the effect of batch size/learning rate on model training in tutorial 4 and 7. So please give me the benefit of the doubt for now. 
   
Please note that this practice of training in mini-batches is not just for deep neural networks, but for any model with 
stochastic gradient descent based implementation, such as SVM, random forest and GBM. If you have not come across mini batch 
based training in sklearn tutorials, that's most likely because these examples are for purposes of illustration and relies 
on a small training/testing dataset. 

## How do we break down data into minibatches for training etc.?  
   First, we need break our data into two parts, training and validation. (Note: this is an incomplete breakdown. For ML
   best practice, we must have train:validation:test. Better still is to have the test dataset from a completely different 
   dataset than the train/validation. We will show how the proper train/val/test split is done in tutorial 4 and onward.) 
   Here I choose to use *kFold* from *sklearn.model_selection*:
            
        splits = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, test_idx in splits.split(X,y):
                X_train, X_test = torch.tensor(X[train_idx], dtype = torch.long), torch.tensor(X[test_idx], dtype = torch.long)
                y_train, y_test = torch.tensor(y[train_idx], dtype = torch.long), torch.tensor(y[test_idx], dtype = torch.long)
                X_lens_train, X_lens_test = torch.tensor(X_lens[train_idx], dtype=torch.float), torch.tensor(X_lens[test_idx], dtype = torch.float)
   
   This is one option but not necessarily the only option. You may also consider *sklearn.model_selection.train_test_split*, 
   or torch.random_split. sklearn.model_selection.stratefiedKFold cannot be used here, as our target is a sequence instead of a discrete number, making stratification non-trivial:  
    
   And then we separate the data into small batches using pytorch's *DataLoader*:

        test_dataset = TensorDataset(X_test, y_test, X_lens_test)
        train_dataset = TensorDataset(X_train, y_train, X_lens_train) 
   
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
        val_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)
   
   (Note: you could also try *BucketIterator* from *torchtext*.)

   We also need to monitor our validation vs. training loss over time to make sure that we are not overfitting to the training data: 
        
        for epoch in range(n_epochs):
            train_losses = []
            val_losses = []
            for X_batch, y_batch, X_lens_batch in train_loader:
                optimizer.zero_grad()
                ypred_batch = model(X_batch, X_lens_batch)

                # flatten y_batch and ypred_batch
                y_batch = y_batch.view(batch_size * seq_len)
                ypred_batch = ypred_batch.view(batch_size * seq_len, -1)

                loss = loss_fn(ypred_batch.view(batch_size*seq_len, -1), y_batch.view(batch_size * seq_len))
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

After training is done, we can visualize our training vs. validation loss in the following [plot](github.com/rantsandruse_pytorch_lstm_02minibatch/blob/main/output/train_vs_val_loss.png): 

## How do we initialize hidden state? 
   In [tutorial 1](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/README.md), you may have noticed that we 
   did not provide input to the initial hidden state of our LSTM network (see main_v1.py(https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_v1.py)):

     lstm_out, (h, c) = self.lstm(embeds.view(len(sentence), 1, -1))

   While in this tutorial, we drew the hidden state from a random uniform distribution using torch.rand and then feed it into our LSTM network:

     lstm_out, self.hidden = self.lstm(embeds, self.hidden)

   At this point you might be asking a couple of questions:
   **First, What was the initial hidden state for our LSTM network in tutorial 1 (I don't remember parsing it in...)?**

   This has a simple answer: If you don't parse in hidden state, it is set to zero by default. 
      
   **And shall we initialize our hidden state randomly or simply set them to zeros**?

   This may not have a simple answer. In general, there are three ways to initialize the hidden state 
   of your LSTM (or RNN network): zero initialization, random initialization, train the initial hidden state as a variable, 
   or some combination of these three options. Each of these methods have its pros and cons. This [blog post](https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html) 
   written by Silviu Pitis provides an excellent explanation (plus experiments) on these options, and I will provide a TL;DR (with some paraphrasing) as below: 
      
   a. Zero initialization is the default initialization method provided by torch.nn.LSTM, and it is usually good enough for 
      seq2seq tasks. This initial zero state is arbitrary, but as the network propagates over a long sequence, the impact of this 
      arbitrary initial state is mitigated over steps and almost eliminated by the end.  However, zero initialization may not be a good idea if 
      the training text contains many short sentences. As the ratio of state resets to total samples increase, the model 
      becomes increasing tuned to zero state, which leads to overfitting.

   b. Random initialization is oftentimes recommended, to combat the aforementioned overfitting problem. The additional noise introduced by random 
      initialization makes the model less sensitive to the initialization and thus less likely to overfit. Note that it can also be combined with the next method. 

   c. Learn a default initial hidden state: If we have many samples requiring a state reset for each of them, such as 
      in a sentiment analysis/sequence classification problem, it makes sense to learn a default initial state. But
      if there are only a few long sequences with a handful of state resets, then learning a default state is prone to overfitting as well. 
      
   d. Silviu used PTB dataset with different initialization methods described above, and made a number of corroborative observations. he showed that: 
        i. All non-zero state initializations sped up training and improved generalization.
        ii. Training the initial state as a variable was more effective than using a noisy zero-mean initial state.
        iii. Adding noise to a variable initial state provided only marginal benefit.
        iv. These non-zero state initializations will only really be useful for datasets that have many naturally-occuring state resets.

   In our case, we are working with a tiny toy dataset, so it doesn't matter much which initialization we use. But ultimately we want to 
   build a sentiment classifier for IMDB reviews, therefore either b or c would be more appropriate. We implemented b in the code: 

        h_0 = torch.randn(1, self.batch_size, self.hidden_dim)
        c_0 = torch.randn(1, self.batch_size, self.hidden_dim)   

   I will leave it out as an exercise to implement method c, i.e. train your initial hidden state as a model parameter (*Hint: you need to add one or two class parameters in your init function, and remember to set requires_grad=True. The solution is
   provided as comments in the code*). 

   Now that we know when/which initialization method to use, you might ask :
   ***Why should we initialize the hidden state every time we feed in a new batch, instead of once and for all?***    
   Since each of our sample is an independent piece of text data, i.e. we have a lot of "state resets", there's no 
   benefit in memorizing the hidden state from one batch and pass it onto another. That said, if our samples were all part 
   of a long sequence, then memorizing the last hidden state will likely be informative for the next training batch.
   
   Last but not least, we've been discussing the initialization of hidden state, which is **very different** from the initialization of the weights of 
   the LSTM network. For the latter, zero initialization is a very bad idea as you are not "breaking the symmetry". In other words, 
   all of the hidden units are getting the same signal and zero signal. You must use random initialization, or other more advanced methods 
   (e.g. Xavier initialization and Kaimin-He initialization). 
   
### How to pad/pack/mask your sequence and why 
  Pytorch tensors are arrays of uniform length, which means that we need to pad all of our sequences to the same length. 
  But padding your sentence without proper downstream processing could have unintended consequences: 
  
  Imagine that you have a training dataset with 99% of sentences under 10 words, and 1% with 100 words or more. For 99% of the time, 
  your model will try to learning paddings, instead of focusing on the actual sequence with meaningful features.  
  
  This is very inefficient. As your LSTM model will waste most of its time learning hidden states for paddings and not the actual sequence. 
  Besides, since we are training a seq2seq model, if you don't explicitly neglect these sequence paddings, then 
  they will show up in your predictions and creep into your loss function and cause significant bias. 
  For these reasons, you need to do the following: 
  1. Pack your sequence. The padding index is set to -1 to enforce the alignment of classes between prediction and ground truth. 
     ((i.e. Both ground truth and prediction uses tag class 0, 1, 2 for the meaningful classes, and cross entropy loss ignores padding class -1 accordingly): 
     
    embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True, enforce_sorted=False)

  2. Feed it into your LSTM model
     
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
  
  Note: we no longer need to reshape the input as we did in tutorial 1.  Since we used the *batch_first=True* option, the 
  input to LSTM here is already (*batch_size, seq_len, hidden_dim*))
      
  3. Pad your sequence back

    lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length = seq_len )
     
  Note: parsing in total_length is a must, otherwise you might run into dimension mismatch.

  4. Last but not least, ask your loss function to ignore the padding, and only use the not ignored elements to calculate 
     the mean loss:
     
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, size_average=True)     

##  Beware  
1. *batch_first* option in nn.LSTM.  
   nn.LSTM function takes in a tensor with the shape (*seq_len, batch_size, hidden_dim*) by default, which is beneficial to tensor operations, but counterintuitive to human users. Switching out 
   batch_first=True allows you parse in a tensor with the shape (*batch_size, seq_len, hidden_dim*). I would recommend the latter to save you a lot of reshaping trouble when parsing mini-batches.
2. We consider hidden states as parameters, but NOT part of the model class parameters (i.e. we did not have *self.hidden*), 
   as we do not need to memorize and reuse them across batches. 

      
## Further Reading 
 1. [What this tutorial was originally based on, including a few fixes/patches discussed above](https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e) 
 2. [How to create minibatches](https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/notebooks/minimal-example-lstm-input.ipynb) 
 3. [How to pad/pack sequence](https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html)
 4. [How to mask/ignore index in cross entropy loss](https://discuss.pytorch.org/t/ignore-index-in-the-cross-entropy-loss/25006/6)
 5. [zero vs. random initialization](https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html)
 5. [Paper on proper initialization of recurrent NN hidden states, See Trick 3](http://www.scs-europe.net/conf/ecms2015/invited/Contribution_Zimmermann_Grothmann_Tietz.pdf)
 6. [Stackflow discussion on hidden state initialization](https://stats.stackexchange.com/questions/224737/best-way-to-initialize-lstm-state)
 7. [Pytorch discussion on hidden state initialization](https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384)
 8. [Stackflow discussion on weight initialization](https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers)



 
    