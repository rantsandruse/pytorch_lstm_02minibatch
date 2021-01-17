'''
This is LSTM Tagger - can train against arbitary targets
What do I want to build next?
What's the improvement from the prior iteration:
1. I want to initialize hidden states - the default h0,c0 are just zero vectors.
2. I will train by batch
3. I will add padding and mask


Reference:
1. https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
2. https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/notebooks/minimal-example-lstm-input.ipynb
3. https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

'''
import torch
from torch import nn

class LSTMTagger(nn.Module):
    '''
    This is v2 of LSTM Tagger.
    The main improvements are invoking batch_first, padding/packing and
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, batch_size=2):
        '''
        embedding_dim: Glove is 300. We are using 6 here.
        hidden_dim: can be anything, usually 32 or 64. We are using 6 here.
        vocab_size: vocabulary size includes an index for padding.
        output_size: We need to exclude the index for padding here.
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # In this case, vocab_size is 9, embedding_dim is 6.
        # hidden dim is also called "number of lstm units"
        # Whenever padding_idx = 0, embedding = 0.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # In this case, vocab_size is 9, embedding_dim is 6.

        # Prepare our model for minibatch based training.
        self.batch_size = batch_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # If you use batch_first=True, Then the input shape will be:
        # (batch_size, seq_len, hidden_dim)
        # If you use batch_first=False, Then the input shape will be:
        # (seq_len, batch_size, hidden_dim)
        # Recommend turning it out, otherwise a pain in the neck.
        # Default number of layers is 1.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        # output_size = tagset_size - 1 to discount padding tag.
        self.hidden2tag = nn.Linear(hidden_dim, output_size)
        print(output_size)

        # Note: the commented line below is what we will do to train our initial hidden state.
        # self.h_0, self.c_0 = self.init_hidden()

    def init_hidden(self):
        '''
        Initiate hidden states.
        '''
        # Shape for hidden state and cell state: num_layers * num_directions, batch, hidden_size
        h_0 = torch.randn(1, self.batch_size, self.hidden_dim)
        c_0 = torch.randn(1, self.batch_size, self.hidden_dim)

        # The Variable API is now semi-deprecated, so we use nn.Parameter instead.
        # Note: For Variable API requires_grad=False by default;
        # For Parameter API requires_grad=True by default.
        h_0 = nn.Parameter(h_0, requires_grad=True)
        c_0 = nn.Parameter(c_0, requires_grad=True)

        return (h_0, c_0)

    def forward(self, sentences, X_lengths):
        '''
        Parameters
        ----------
        sentences: padded sentences tensor. Each element of the tensor is an array of words.
        X_lengths: length of sentence tensor. Each element of the tensor is the original
                   length of the unpadded sentence.

        Returns
        -------

        '''
        # Dimensions of tensors:
        # (Note that seq_len is max length)
        # Shape of embedding (embeds): batch_size, seq_len, hidden_dim
        # Shape of embedding post packing (embeds): batch_size, orig_len, hidden_dim
        # Shape of self.hidden: (num_layers*num_directions, batch_size, hidden_dim)
        # Shape of lstm_out: batch_size, seq_len, hidden_dim
        # Shape of tag_scores: batch_size, 1
        hidden_0 = self.init_hidden()
        batch_size, seq_len = sentences.size()
        embeds = self.word_embeddings(sentences)

        # We need to reshape the batch from
        # This is the shape wanted according to lstm user manual (Need to understand why?)
        # print("embedding shape:", embeds.shape)

        # By setting batch_first=true, we are outputting a tensor of:
        # (batch_size, seq_len, input_size)
        # instead of the default:
        # (seq_len, batch_size, input_size)
        # Having batch_size first is more intuitive to human, while having seq_len as the first
        # dimension makes tensor operations easier.
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True, \
                                                         enforce_sorted=False)
        # Note: we no longer need to reshape the input: As we used batch first, the input to LSTM
        # here is already (batch_size, seq_len, hidden_dim)
        # The original code is:
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # We choose not to save self.hidden as we re-initialize the hidden state for the new batch to random anyways.
        # Note: the commented line below is what we will do if we want to train our own initial state.
        # lstm_out, _ = self.lstm(embeds, (self.h_0, self.c_0))
        lstm_out, _ = self.lstm(embeds, hidden_0)

        # Note: parsing in total_length is a must, otherwise you might run into dimension mismatch.
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, \
                                                             total_length = seq_len )

        #Take batch size into account
        tag_scores = self.hidden2tag(lstm_out.view(batch_size* seq_len, -1))

        return tag_scores
