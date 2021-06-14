import torch
import torch.nn as nn
import math

# an implementation of several parametrizations for orthogonal transformations
# no bias!


# cayley


# householder
# TODO normalize sometimes to keep <v, v> approx equal to one for numerical stability
class OrthogonalHouseholder(nn.Module):
    def __init__(self, sz, bias=True):
        super(OrthogonalHouseholder, self).__init__()
        self.sz = sz
        self.bias = bias

        self.A = nn.Parameter(torch.empty((sz, sz)))
        self.b = nn.Parameter(torch.empty(sz)) if bias else 0.
        self.reset_parameters()

    def reset_parameters(self):
        # i choosed initialitazion at random. Another one might be much better
        with torch.no_grad():
            self.A.normal_(0, math.sqrt(2 / self.sz))
            if self.bias:
                self.b.fill_(0.)

    def forward(self, x):
        norms_sq = torch.einsum("ij,ij->i", self.A, self.A)
        for i in range(self.sz):
            x = x - 2 * self.A[i] * (x @ self.A[i].unsqueeze(1)) / norms_sq[i]
        return x + self.b

# should be faster on gpu? or at least might be faster in recurrent net if caching added
class OrthogonalHouseholderAlternative(nn.Module):
    def __init__(self, sz, bias=True):
        super(OrthogonalHouseholderAlternative, self).__init__()
        self.sz = sz
        self.bias = bias

        self.A = nn.Parameter(torch.empty((sz, sz)))
        self.b = nn.Parameter(torch.empty(sz)) if bias else 0.
        self.reset_parameters()

    def reset_parameters(self):
        # i choosed initialitazion at random. Another one might be much better
        with torch.no_grad():
            self.A.normal_(0, math.sqrt(2 / self.sz))
            if self.bias:
                self.b.fill_(0.)

    # this part can be cached between updates
    # sz^3
    def _forward_precalc(self):
        B = self.A @ self.A.T
        self.diag = torch.diag(B)
        self.p = self.A.clone() # no detach!
        for i in range(self.sz - 1):
            self.p[i+1:] = self.p[i+1:].clone() - (2 * B[i, i+1:] / self.diag[i+1:]).unsqueeze(1) * self.p[i].clone()

    def forward(self, x):
        self._forward_precalc()
        B = x @ self.A.T
        x = x - ((2 * B / self.diag) @ self.p)
        return x + self.b


# exponential



