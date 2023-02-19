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


class TriConv1d(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    self.prenet = PreNet(in_dim, 32, 32)
    self.convs = nn.Sequential(
      nn.Conv1d(32, 32, 15, 1, 7),
      nn.ReLU(),
      nn.Conv1d(32, 32, 15, 1, 7),
      nn.ReLU(),
      nn.Conv1d(32, 32, 15, 1, 7),
      nn.ReLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.prenet(x)
    x = self.convs(x.transpose(1, 2))
    return x.transpose(1, 2)


class LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.prenet = PreNet(hp.OUTPUT_DIM, 32, 32)
    self.lstm = nn.LSTM(32 + 32, 64, batch_first=True)
    self.proj = nn.Linear(64, hp.OUTPUT_DIM, bias=False)

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = self.prenet(y)
    x, _ = self.lstm(torch.cat((x, y), dim=-1))   # [64, 191, 96]
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, xs: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xs.size(0), hp.OUTPUT_DIM, device=xs.device)
    h = torch.zeros(1, xs.size(0), 64, device=xs.device)
    c = torch.zeros(1, xs.size(0), 64, device=xs.device)

    ys = []
    for x in torch.unbind(xs, dim=1):
      y = self.prenet(y)
      x = torch.cat((x, y), dim=1).unsqueeze(1)
      x, (h, c) = self.lstm(x, (h, c))
      y = self.proj(x).squeeze(1)
      ys.append(y)
    return torch.stack(ys, dim=1)


class TriLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.prenet = PreNet(hp.OUTPUT_DIM, 32, 32)
    self.lstm1 = nn.LSTM(32 + 32, 64, batch_first=True)
    self.lstm2 = nn.LSTM(64, 64, batch_first=True)
    self.lstm3 = nn.LSTM(64, 64, batch_first=True)
    self.proj = nn.Linear(64, hp.OUTPUT_DIM, bias=False)

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = self.prenet(y)
    x, _ = self.lstm1(torch.cat((x, y), dim=-1))
    res = x
    x, _ = self.lstm2(x)
    x = res + x
    res = x
    x, _ = self.lstm3(x)
    x = res + x
    return self.proj(x)

  @torch.inference_mode()
  def generate(self, xs: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xs.size(0), hp.OUTPUT_DIM, device=xs.device)
    h1 = torch.zeros(1, xs.size(0), 64, device=xs.device)
    c1 = torch.zeros(1, xs.size(0), 64, device=xs.device)
    h2 = torch.zeros(1, xs.size(0), 64, device=xs.device)
    c2 = torch.zeros(1, xs.size(0), 64, device=xs.device)
    h3 = torch.zeros(1, xs.size(0), 64, device=xs.device)
    c3 = torch.zeros(1, xs.size(0), 64, device=xs.device)

    ys = []
    for x in torch.unbind(xs, dim=1):
      y = self.prenet(y)
      x = torch.cat((x, y), dim=1).unsqueeze(1)
      x1, (h1, c1) = self.lstm1(x, (h1, c1))
      x2, (h2, c2) = self.lstm2(x1, (h2, c2))
      x = x1 + x2
      x3, (h3, c3) = self.lstm3(x, (h3, c3))
      x = x + x3
      y = self.proj(x).squeeze(1)
      ys.append(y)
    return torch.stack(ys, dim=1)


''' Model '''
class CRNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.embd_weekday = nn.Embedding(7,  8)  #[24,719,1] → [24,719,8]
    self.embd_hour    = nn.Embedding(24, 24) #[24,719,1] → [24,719,24]
    self.encoder      = TriConv1d(hp.INPUT_DIM)        #24+8+5=37
    self.decoder      = LSTM() 
    #self.decoder      = TriLSTM()

  def forward(self, w, h, d_x, d_y):          #训练时使用，告知正确值
    w = self.embd_weekday(w)                  #[B=24, T=719, D=8]
    h = self.embd_hour(h)                     #[B=24, T=719, D=24]
    x = torch.concat([w, h, d_x], axis=-1)    #[B=24, T=719, D=37]
    
    x = self.encoder(x)
    return self.decoder(x, d_y)               # d_y.shape == [24,719,5]

  @torch.inference_mode()                     #验证/测试时使用，不能告知正确值
  def generate(self, w, h, d):
    w = self.embd_weekday(w)
    h = self.embd_hour(h)
    x = torch.concat([w, h, d], axis=-1)

    x = self.encoder(x)
    return self.decoder.generate(x)
