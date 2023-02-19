#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/15 

import torch
import torch.nn as nn

import hparam as hp


''' Module '''
class PreNet(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, output_size),
      nn.ReLU(),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

# downsample x8
class TriConv1d(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    self.prenet = PreNet(in_dim, 32, 128)
    self.convs = nn.Sequential(
      nn.Conv1d(128, 128, 15, 2),
      nn.ReLU(),
      nn.Conv1d(128, 128, 15, 2),
      nn.ReLU(),
      nn.Conv1d(128, 128, 15, 2),
      nn.ReLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.prenet(x)
    x = self.convs(x.transpose(1, 2))
    return x.transpose(1, 2)


class DuoLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.prenet = PreNet(hp.OUTPUT_DIM, 128, 128)
    self.lstm1 = nn.LSTM(128 + 128, 128, batch_first=True)
    self.lstm2 = nn.LSTM(128, 128, batch_first=True)
    self.proj = nn.Linear(128, hp.OUTPUT_DIM, bias=False)

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = self.prenet(y)
    x, _ = self.lstm1(torch.cat((x, y), dim=-1))
    res = x
    x, _ = self.lstm2(x)
    x = res + x
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, xs: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xs.size(0), hp.OUTPUT_DIM, device=xs.device)
    h1 = torch.zeros(1, xs.size(0), 128, device=xs.device)
    c1 = torch.zeros(1, xs.size(0), 128, device=xs.device)
    h2 = torch.zeros(1, xs.size(0), 128, device=xs.device)
    c2 = torch.zeros(1, xs.size(0), 128, device=xs.device)

    ys = []
    for x in torch.unbind(xs, dim=1):
      y = self.prenet(y)
      x = torch.cat((x, y), dim=1).unsqueeze(1)
      x1, (h1, c1) = self.lstm1(x, (h1, c1))
      x2, (h2, c2) = self.lstm2(x1, (h2, c2))
      x = x1 + x2
      y = self.proj(x).squeeze(1)
      ys.append(y)
    return torch.stack(ys, dim=1)


''' Model '''
class CRNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.embd_weekday = nn.Embedding(7,  hp.EMBED_WEEKDAY_DIM)
    self.embd_hour    = nn.Embedding(24, hp.EMBED_HOUR_DIM)
    self.embd_features = nn.ModuleList([
      nn.Embedding(hp.QT_N_BIN, hp.FATURE_DIM) for _ in range(hp.N_FEATURES)
    ])
    self.encoder      = TriConv1d(hp.INPUT_DIM)
    self.decoder      = DuoLSTM()

  def forward(self, w, h, d_x, d_y):
    w = self.embd_weekday(w)                  # [B=32, T=191, D=8]
    h = self.embd_hour(h)                     # [B=32, T=191, D=24]
    e_d_x = [self.embd_features[i](ft) for i, ft in enumerate(d_x)]
    x = torch.concat([w, h, *e_d_x], axis=-1) # [B=32, T=191, D=35]

    e_d_y = self.embd_features[0](d_y)
    
    x = self.encoder(x)
    return self.decoder(x, e_d_y)             # d_y.shape == [32, 191, 3]

  @torch.inference_mode()
  def generate(self, w, h, d):
    w = self.embd_weekday(w)
    h = self.embd_hour(h)
    x = torch.concat([w, h, d], axis=-1)

    x = self.encoder(x)
    return self.decoder.generate(x)
