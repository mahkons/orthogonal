import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import time
import math

from orthogonal import OrthogonalHouseholder, OrthogonalHouseholderAlternative

# test linear layers on MNIST

device = torch.device("cuda")

def normalize(module):
    if module is OrthogonalHouseholder:
        module.A /= torch.sqrt((module.A * module.A).sum(dim=1))

class Classfier(nn.Module):
    def __init__(self, in_sz, out_sz, hidden_sz, hidden_layers_factory, hiddens_num):
        super(Classfier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_sz, hidden_sz),
            nn.ReLU(),
            *sum([hidden_layers_factory(hidden_sz) for i in range(hiddens_num)], []),
            nn.Linear(hidden_sz, out_sz),
        )

    # returns logits of softmax
    def forward(self, x):
        return self.seq(x)

class MyActivation(nn.Module):
    def __init__(self, coeff):
        super(MyActivation, self).__init__()
        self.coeff = coeff
        self.a = 1 / math.sqrt(2 * self.coeff**2 - self.coeff**4)

    def forward(self, x):
        return torch.where(x > 0, self.coeff * x, x / (self.a * self.coeff))

class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(-x**2)


def simple_linear_factory(sz):
    return [nn.Linear(sz, sz, bias=True), nn.ReLU()]

def orthogonal_householder_factory(sz):
    return [OrthogonalHouseholder(sz, bias=True), nn.ReLU()]

def orthogonal_householder_myactivation_factory(sz):
    return [OrthogonalHouseholder(sz, bias=True), MyActivation(1.2)]

def orthogonal_householder_expactivation_factory(sz):
    return [OrthogonalHouseholder(sz, bias=True), ExpActivation()]

def orthogonal_householder_alternative_factory(sz):
    return [OrthogonalHouseholderAlternative(sz, bias=True), nn.ReLU()]

# any matrix with determinant equal to +-1 can be represented
#   as a product of two orthogonal matrices
def orthogonal_householder_double_factory(sz):
    return [OrthogonalHouseholder(sz), OrthogonalHouseholder(sz, bias=True), nn.ReLU()]

def orthogonal_householder_alternative_double_factory(sz):
    return [OrthogonalHouseholderAlternative(sz), OrthogonalHouseholderAlternative(sz, bias=True), nn.ReLU()]


def train(model, train_data, test_data, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    it = 0
    for epoch in range(epochs):
        train_loss = 0.
        train_cnt = 0.
        for x, y in train_loader:
            it += 1
            x, y = torch.flatten(x.to(device), start_dim=1), y.to(device)
            logits = model(2 * x - 1)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_cnt += len(y)

            if it % 100 == 0:
                model.apply(normalize)

        sum_loss, cnt, cnt_correct = 0, 0, 0
        for x, y in test_loader:
            x, y = torch.flatten(x.to(device), start_dim=1), y.to(device)
            with torch.no_grad():
                logits = model(2 * x - 1)
                loss = F.cross_entropy(logits, y, reduction='sum')
                sum_loss += loss
                cnt += len(y)
                cnt_correct += (torch.argmax(logits, dim=1) == y).sum()

        cur_time = time.time()
        print("Epoch {} Time {} TrainLoss {} TestLoss {} TestAccuracy {}".format(
            epoch, cur_time - start_time, train_loss / train_cnt, sum_loss / cnt, cnt_correct / cnt))



if __name__ == "__main__":
    transform = T.ToTensor()
    train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    #  model = Classfier(28 * 28, 10, 64, simple_linear_factory, 3).to(device)
    #  model = Classfier(28 * 28, 10, 64, simple_linear_factory, 6).to(device)
    #  model = Classfier(28 * 28, 10, 16, orthogonal_householder_abs_factory, 30).to(device)
    #  model = Classfier(28 * 28, 10, 16, orthogonal_householder_alternative_factory, 3).to(device)
    model = Classfier(28 * 28, 10, 4, orthogonal_householder_myactivation_factory, 100).to(device)

    train(model, train_data, test_data, 100)

