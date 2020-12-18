import torch.nn as nn
from LSTMCell import LSTMCell

class MultiLayerLSTM(nn.Module):
    # Multilyaer LSTM is just the stacking of NaiveLSTM. 可以做并行计算，但是我这里还是一层一层算吧
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(MultiLayerLSTM, self).__init__()
        self.num_layers = num_layers
        self.LSTMs = nn.ModuleList()
        self.LSTMs.append(LSTMCell(input_size, hidden_size, batch_first=batch_first))
        for i in range(num_layers - 1):
            self.LSTMs.append(LSTMCell(hidden_size, hidden_size, batch_first=batch_first))

    def forward(self, x, state):
        (h0s, c0s) = state
        for i in range(self.num_layers):
            x, _ = self.LSTMs[i](x, (h0s[i].unsqueeze(0), c0s[i].unsqueeze(0)))
        return x, _