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

# orthogonally regualarization loss returning function added
class MyLinear(nn.Module):
    def __init__(self, in_sz, out_sz, bias=True):
        super(MyLinear, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.bias = bias

        self.W = nn.Parameter(torch.empty((in_sz, out_sz)))
        self.b = nn.Parameter(torch.empty(1, out_sz)) if bias else 0.
        self.reset_parameters()

    def forward(self, x):
        return x @ self.W + self.b

    def regularization(self):
        return torch.linalg.norm(self.W @ self.W.T - torch.eye(self.in_sz, device=self.W.device)) + torch.linalg.norm(self.W.T - self.W - torch.eye(self.out_sz, device=self.W.device))

    def reset_parameters(self):
        # i choosed initialitazion at random. Another one might be much better
        with torch.no_grad():
            self.W.normal_(0, math.sqrt(4 / (self.in_sz + self.out_sz)))
            if self.bias:
                self.b.fill_(0.)

# exponential



