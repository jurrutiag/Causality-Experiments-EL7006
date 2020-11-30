import torch
import torch.nn as nn

class Marginal(nn.Module):
    def __init__(self, N, dtype=None):
        super(Marginal, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros(N, dtype=dtype))
    
    def forward(self, inputs):
        # log p(A) / log p(B)
        cste = torch.logsumexp(self.w, dim=0)
        return self.w[inputs.squeeze(1)] - cste

class Conditional(nn.Module):
    def __init__(self, N, dtype=None):
        super(Conditional, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))
    
    def forward(self, inputs, conds):
        # log p(B | A) / log p(A | B)
        conds_ = conds.squeeze(1)
        cste = torch.logsumexp(self.w[conds_], dim=1)
        return self.w[conds_, inputs.squeeze(1)] - cste

class BiConditional(nn.Module):
    def __init__(self, N, dtype=None):
        super(Conditional, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N, N), dtype=dtype))
    
    def forward(self, inputs, conds1, conds2):
        # log p(B | A, C)
        conds1_ = conds1.squeeze(1)
        conds2_ = conds2.squeeze(1)
        cste = torch.logsumexp(self.w[conds1_, conds2_], dim=1)
        return self.w[conds1_, conds2_, inputs.squeeze(1)] - cste
