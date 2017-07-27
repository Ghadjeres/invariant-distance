from itertools import islice

import torch
from deepPermutations.losses import crossentropy_loss, accuracy
from deepPermutations.sequential_model import SequentialModel
from torch.autograd import Variable
from tqdm import tqdm


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

        self.optimizer.zero_grad()

        mce_loss, reg, acc = self.model.loss_functions(next_element,
                                                       reg_norm)
        loss = mce_loss + self.lambda_reg * reg

        # backward pass and step
        if train:
            loss.backward()
            self.optimizer.step()

        # compute mean loss and accuracy
        return (variable2float(mce_loss),
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
