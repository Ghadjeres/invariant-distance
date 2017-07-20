from itertools import islice

import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

from deepPermutations.sequential_model import SequentialModel


def variable2float(v: Variable):
    return float(v.data.cpu().numpy())

def plot_init(res_size):
    """

    :param res_size: result size
    :type res_size:
    :return:
    :rtype:
    """
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(res_size, sharex=True)
    x = []
    ys = [[] for _ in range(res_size)]
    ys_val = [[] for _ in range(res_size)]
    fig.show()
    return axarr, fig, plt, x, ys, ys_val


def plot_res(axarr, epoch_index, fig, plt, res, res_size, res_val, x,
             ys, ys_val):
    x.append(epoch_index)
    for res_index in range(res_size):
        y = res[res_index]
        ys[res_index].append(y)

        y_val = res_val[res_index]
        ys_val[res_index].append(y_val)

        axarr[res_index].plot(x, ys[res_index], 'r-',
                              x, ys_val[res_index], 'r--')
    fig.canvas.draw()
    plt.pause(0.001)


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


class ModelManager:
    def __init__(self, model: SequentialModel, lr=1e-3, lambda_reg=1e-5):
        self.model = model
        self.model.cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr)
        self.lambda_reg = lambda_reg

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator,
                              train=True, reg_norm=0):

        mean_mce_loss = 0
        mean_accuracy = 0
        mean_reg = 0
        if train:
            self.model.train()
        else:
            self.model.eval()
        for sample_id, next_element in tqdm(
                enumerate(islice(generator, batches_per_epoch))):
            mce_loss, grad_reg, acc = self.loss_and_acc(
                next_element,
                reg_norm,
                train=train)

            mean_mce_loss += mce_loss
            mean_reg += grad_reg
            mean_accuracy += acc

        return (mean_mce_loss / batches_per_epoch,
                mean_reg / batches_per_epoch,
                mean_accuracy / batches_per_epoch)

    def loss_and_acc(self, next_element, reg_norm, train):

        # to cuda
        next_element_cuda = [
            Variable(tensor.cuda())
            for tensor in next_element
        ]

        input_1, input_2, first_note, output = next_element_cuda

        self.optimizer.zero_grad()

        # forward pass
        inputs = (input_1, input_2)
        weights, softmax, diff = self.model.forward(inputs, first_note)

        # mce_loss
        mce_loss = crossentropy_loss(weights, output)

        # compute loss
        loss = mce_loss

        # regularization
        if reg_norm is not None:
            reg = torch.norm(diff, p=reg_norm, dim=1).mean()
            loss += self.lambda_reg * reg

        # backward pass and step
        if train:
            loss.backward()
            self.optimizer.step()

        # accuracy
        acc = accuracy(weights, output)

        # compute mean loss and accuracy
        return (variable2float(mce_loss),
                # variable2float(grad_reg),
                variable2float(reg),
                acc)

    def train_model(self,
                    batch_size,
                    batches_per_epoch,
                    num_epochs,
                    plot=False,
                    save_every=10,
                    reg_norm=None):
        generator_train = self.model.generator(phase='train',
                                               batch_size=batch_size)
        generator_val = self.model.generator(phase='test',
                                             batch_size=batch_size)

        res_size = 3
        if plot:
            axarr, fig, plt, x, ys, ys_val = plot_init(res_size=res_size)

        for epoch_index in range(num_epochs):
            # train
            res = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch,
                generator=generator_train,
                train=True,
                reg_norm=reg_norm)

            # eval
            res_val = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch // 10,
                generator=generator_val,
                train=False,
                reg_norm=reg_norm)

            # plot
            if plot:
                plot_res(axarr, epoch_index, fig, plt, res, res_size,
                         res_val, x, ys, ys_val)

            print(f"{',  '.join(map(str, res_val))}\n")

            if (epoch_index + 1) % save_every == 0:
                self.save()
                print('Model saved')
