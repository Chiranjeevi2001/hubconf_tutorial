import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.optim as optim

mnist_train = datasets.MNIST(
  root='data',
  train = True,
  download = True,
  transform = ToTensor()
)

batch_size = 64

train_dl = DataLoader(mnist_train, batch_size = batch_size)

class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super(LSTM, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim

    self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)

    self.fc = nn.Linear(hidden_dim, output_dim)


  def forward(self, x):
    h0 = Variable(torch.zeros(layer_dim, x.size(0), hidden_dim))

    out, hn = self.lstm(x, h0)

    out = self.fc(out[:, -1, :])

    return out

input_dim = 28
output_dim = 10
hidden_dim = 100
layer_dim = 1
seq_dim = 28

model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.005)

def train_model(num_epochs = 10, model = model, train_dl = train_dl, optimizer = optimizer, loss = criterion):
  for epoch in num_epochs:
    for i , (X, y) in enumerate(train_dl):
      X = Variable(X.view(-1, seq_dim, input_dim))
      y = Variable(y)
      out = model(X)

      loss = loss(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if(i%500 == 0):
        print(f"Epoch:{epoch}, i:{i}, loss:{loss.data}")
