import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from typing import Tuple
from torch.nn import Parameter
import math
import torch.nn.init as init
import time


class NaiveLSTM(nn.Module):
    """Naive LSTM like nn.LSTM"""

    # Naive LSTM has only 1 layer

    def __init__(self, input_size, hidden_size, batch_first):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_fisrt = batch_first

        # 输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, state):
        #       ->用来提示该函数返回值的数据类型
        """Forward
        Args:
            inputs: (batch_size, seq_size, feature_size)
            state: h0,c0 (num_layer, batch_size, hidden_size)
        """
        if (self.batch_fisrt):
            batch_size, seq_size, _ = inputs.size()

        if state is None:
            h = torch.zeros(1, batch_size, hidden_size)  # num_layer is 1
            c = torch.zeros(1, batch_size, hidden_size)
        else:
            (h, c) = state
            h = h
            c = c

        # squeeze过后，h,c(batch_size,hidden_size)

        hidden_seq = []

        # seq_size = 28
        for t in range(seq_size):
            x = inputs[:, t, :]

            # input gate
            i = torch.sigmoid(x @ self.w_ii.t() + self.b_ii.t() + h @ self.w_hi.t() +
                              self.b_hi.t())

            # forget gate
            f = torch.sigmoid(x @ self.w_if.t() + self.b_if.t() + h @ self.w_hf.t() +
                              self.b_hf.t())

            # cell
            g = torch.tanh(x @ self.w_ig.t() + self.b_ig.t() + h @ self.w_hg.t()
                           + self.b_hg.t())
            # output gate
            o = torch.sigmoid(x @ self.w_io.t() + self.b_io.t() + h @ self.w_ho.t() +
                              self.b_ho.t())

            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            c = c_next
            h = h_next
            hidden_seq.append(h)
        hidden_seq = torch.cat(hidden_seq, dim=2).reshape(batch_size, seq_size, hidden_size)
        return hidden_seq, (h, c)


class MultiLayerLSTM(nn.Module):
    # Multilyaer LSTM is just the stacking of NaiveLSTM. 可以做并行计算，但是我这里还是一层一层算吧
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(MultiLayerLSTM, self).__init__()
        self.num_layers = num_layers
        self.LSTMs = nn.ModuleList()
        self.LSTMs.append(NaiveLSTM(input_size, hidden_size, batch_first=True))
        for i in range(num_layers - 1):
            self.LSTMs.append(NaiveLSTM(hidden_size, hidden_size, batch_first=True))

    def forward(self, x, state):
        (h0s, c0s) = state
        for i in range(self.num_layers):
            x, _ = self.LSTMs[i](x, (h0s[i].unsqueeze(0), c0s[i].unsqueeze(0)))
        return x, _


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = MultiLayerLSTM(input_size, hidden_size, num_layers,
                                   batch_first=True)  # batch_first 为 True则输入输出的数据格式为 (batch, seq, feature)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x (100,28,28)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            device)  # 当num_layer为1时，h0 (1,100,128) 即为 (num_layer, batch_size, hidden_size) 为batch_size的每一个输入初始化一个h0 c0? 不过这里没区别哈，都是0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)  (100,28,128)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.reshape(-1, sequence_length, input_size).to(device) # 这里的-1应该是表示待定，这里的意思就是把tensor转换为 a * 28 * 28

        # break
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

end = time.time()
print(end - start)
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')