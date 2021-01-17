'''
Util functions to preprocess sequences
'''
import torch
import numpy as np


def seq_to_embedding(seq, to_ix):
    '''
    This is a good entry point for passing in different kinds of embeddings and
    :param seq: sequence of words
    :param to_ix: embedding lib
    :return:
    '''
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def seqs_to_dictionary(training_data: list):
    '''
    Parameters
    ----------
    training_data: training data as a list of tuples.

    Returns
    -------
    word_to_ix: a dictionary mapping words to indices
    tag_to_ix: a dictionary mapping tags to indices
    '''
    # Prepare for padding; need to change count to 1 as well.
    word_to_ix = {'<PAD>':0}
    tag_to_ix = {'<PAD>':0}
    count1 = count2 = 1

    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = count1
                count1 += 1
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = count2
                count2 += 1
    return word_to_ix, tag_to_ix

# Note: we are stealing this from tf.keras.preprocessing.sequence, to avoid having an additional dependency for tensorflow.
# It's perfectly fine if you want to install tf and call the function directly from there.


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    Arguments:
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float, padding value.
    Returns:
        x: Numpy array with shape `(len(sequences), maxlen)`
    Raises:
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
