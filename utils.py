import torch
def getLL1Mean(w):
    '''
    应该会很慢
    :param w: matrix tensor
    :return:
    '''
    sum = torch.sum(torch.abs(w))
    n = w.shape[0] * w.shape[1]
    return sum/n