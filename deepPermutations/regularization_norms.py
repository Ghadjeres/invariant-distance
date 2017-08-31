import torch
from torch.autograd import Variable


def regularization_norm_from_name(reg_norm_name):
    if reg_norm_name is not None:
        if reg_norm_name == 'l1':
            return lambda x: torch.abs(x).sum(1).mean()
        elif reg_norm_name == 'l2':
            return lambda x: torch.norm(x, p=2, dim=1).mean()
        else:
            NotImplementedError
    else:
        return lambda x: Variable(torch.zeros(1).cuda())


