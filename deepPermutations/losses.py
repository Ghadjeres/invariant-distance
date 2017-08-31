import torch
from torch import nn


def crossentropy_loss(output_seq, targets_seq):
    """
    :param output_seq: (seq_length, batch_size, num_features)
    of weights for each features
    :type output_seq:
    :param targets_seq: (batch_size, seq_length)
    of class indexes (between 0 and num_features -1)
    :type targets_seq:
    :return:
    :rtype:
    """
    targets_seq = torch.transpose(targets_seq, 0, 1)
    assert output_seq.size()[:-1] == targets_seq.size()
    seq_length, _, _ = output_seq.size()
    cross_entropy = nn.CrossEntropyLoss(size_average=True)
    sum = 0
    for t in range(seq_length):
        ce = cross_entropy(output_seq[t], targets_seq[t])
        sum += ce
    return sum


def accuracy(output_seq, targets_seq):
    """

    :param output_seq: (seq_length, batch_size, num_features)
    of weights for each features
    :type output_seq:
    :param targets_seq: (batch_size, seq_length)
    of class indexes (between 0 and num_features -1)
    :type targets_seq:
    :return:
    :rtype:
    """
    targets_seq = torch.transpose(targets_seq, 0, 1)
    assert output_seq.size()[:-1] == targets_seq.size()
    seq_length = output_seq.size()[0]
    batch_size = output_seq.size()[1]
    sum = 0
    for t in range(seq_length):
        max_values, max_indices = output_seq[t].max(1)
        correct = max_indices[:, 0] == targets_seq[t]
        sum += correct.data.sum() / batch_size

    return sum / seq_length
