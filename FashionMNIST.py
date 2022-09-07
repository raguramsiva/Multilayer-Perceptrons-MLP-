import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms


def get_labels(labels):
    """Return text labels for Fashion-MNIST labels."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data(batch_size, resize=None):
    """Download Fashion-MNIST training and test datasets."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


def init_weights(m):
    """Initialize parameter weights."""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, y):  # @save
    """Return number of correct labels."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """Return accuracy of model."""
    if isinstance(net, torch.nn.Module):
        net.eval()
    accum = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            accum.add(accuracy(net(X), y), y.numel())
    return accum[0] / accum[1]


class Accumulator:

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def epoch_training(net, train_iter, loss, updater):
    """Train model for one epoch."""
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    training_loss = metric[0] / metric[2]
    training_accuracy = metric[1] / metric[2]

    return training_loss, training_accuracy


def training(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train model for num_epochs many epochs."""
    for epoch in range(num_epochs):
        train_metrics = epoch_training(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)


def evaluate_model(net, test_iter):  # @save
    """Evaluate model on subset of testing dataset."""
    for X, y in test_iter:
        break

    accuracy_percentage = accuracy(net(X), y) / len(y)
    print(f'Accuracy of model is {accuracy_percentage:f}.')


if __name__ == "__main__":
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 10))
    net.apply(init_weights)
    batch_size = 256
    num_epochs = 15
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    train_iter, test_iter = load_data(batch_size)
    training(net, train_iter, test_iter, loss, num_epochs, trainer)
    evaluate_model(net, test_iter)
