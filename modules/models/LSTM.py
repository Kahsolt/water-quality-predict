import torch.nn as nn

class LSTM(nn.Module):

  def __init__(self, d_in, d_out, d_hidden=128, n_layers=1):
    super().__init__()

    self.lstm = nn.LSTM(d_in, d_hidden, n_layers)
    self.linear = nn.Linear(d_hidden, d_out)

  def forward(self, x):
    x = x.permute([1, 0, 2])      # [B, L, D] => [L, B, D]
    x, _ = self.lstm(x)           # [L, B, I=1} => [L, B, H]
    x = x[-1, ...]                # last frame of outputs
    x = self.linear(x)            # [B, H] => [B, O]
    return x
