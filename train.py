import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import time
import PIL
import math

from orthogonal import OrthogonalHouseholder, OrthogonalHouseholderAlternative, MyLinear

# test linear layers on MNIST

device = torch.device("cuda")

@torch.no_grad()
def normalize(module):
    if isinstance(module, OrthogonalHouseholder):
        module.A /= torch.linalg.norm(module.A, dim=1)

def get_orthogonal_regularization(module):
    if isinstance(module, MyLinear):
        return module.regularization()
    else:
        return 0.

class RandomInvert(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return PIL.ImageOps.invert(img)
        return img

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
        self.a = 1 / math.sqrt(2 - self.coeff**2)

    def forward(self, x):
        return torch.where(x > 0, self.coeff * x, x / self.a)

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

def orthogonal_householder_double_factory(sz):
    return [OrthogonalHouseholder(sz), OrthogonalHouseholder(sz, bias=True), nn.ReLU()]

def orthogonal_householder_alternative_double_factory(sz):
    return [OrthogonalHouseholderAlternative(sz), OrthogonalHouseholderAlternative(sz, bias=True), nn.ReLU()]

def mylinear_factory(sz):
    return [MyLinear(sz, sz, bias=True), nn.ReLU()]

def mylinear_myactivation_factory(sz):
    return [MyLinear(sz, sz, bias=True), MyActivation(1.2)]


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
        reg_loss = 0.
        train_iter = 0
        for x, y in train_loader:
            it += 1
            train_iter += 1
            x, y = torch.flatten(x.to(device), start_dim=1), y.to(device)

            logits = model(2 * x - 1)
            loss = F.cross_entropy(logits, y)

            rl = torch.zeros(1, device=device)
            for module in model.modules():
                rl += 1e-3 * get_orthogonal_regularization(module)

            optimizer.zero_grad()
            (loss + rl).backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            reg_loss += rl.item()
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
        print("Epoch {} Time {} TrainLoss {} RegLoss {} TestLoss {} TestAccuracy {}".format(
            epoch, cur_time - start_time, train_loss / train_cnt, reg_loss / train_iter, sum_loss / cnt, cnt_correct / cnt))



if __name__ == "__main__":
    transform = T.ToTensor()
    much_regularized = T.Compose([
        T.Pad(padding=4),
        T.CenterCrop((28, 28)),
        T.GaussianBlur(3),
        T.RandomPerspective(),
        T.RandomRotation(degrees=(0, 360)),
        RandomInvert(),
        T.ToTensor()
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=much_regularized, download=True)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    #  model = Classfier(28 * 28, 10, 64, simple_linear_factory, 3).to(device)
    #  model = Classfier(28 * 28, 10, 64, simple_linear_factory, 6).to(device)
    #  model = Classfier(28 * 28, 10, 16, orthogonal_householder_abs_factory, 30).to(device)
    #  model = Classfier(28 * 28, 10, 16, orthogonal_householder_alternative_factory, 3).to(device)
    #  model = Classfier(28 * 28, 10, 64, orthogonal_householder_myactivation_factory, 3).to(device)
    #  model = Classfier(28 * 28, 10, 64, mylinear_factory, 3).to(device)
    model = Classfier(28 * 28, 10, 64, mylinear_myactivation_factory, 64).to(device)

    train(model, train_data, test_data, 50)

