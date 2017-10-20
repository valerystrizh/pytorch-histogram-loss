import torch
import torch.nn as nn

class DropoutShared(nn.Module):
    def __init__(self, p=0.5, use_gpu=True):
        super(DropoutShared, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.use_gpu = use_gpu

    def forward(self, input):
        if self.training:
            index = torch.range(0, input.size()[1] - 1)[torch.Tensor(input.size()[1]).uniform_(0, 1).le(self.p)].long()
            input_cloned = input.clone()
            if self.use_gpu:
                input_cloned[:, index.cuda()] = 0
            else:
                input_cloned[:, index] = 0
            return input_cloned / (1 - self.p)
        else:
            return input

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) + ')'
        
        
class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        input = input.squeeze()
        return input.div(torch.norm(input, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__