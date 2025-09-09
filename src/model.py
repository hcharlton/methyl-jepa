# implement an Enum for the featureset possibilities
# serves as self docs for possibilities
# makes errors more understandable

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
from enum import Enum

class FeatureSet(Enum):
  NUCLEOTIDES = "nucleotides"
  KINETICS = "kinetics"
  ALL = "all"

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.relu(self.conv(x))
    return out

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1):
    super(ResBlock, self).__init__()

    padding1 = (kernel_size - 1) // 2
    padding2 = (kernel_size - 1) // 2

    self.bn1 = nn.BatchNorm1d(in_channels)
    self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding1)
    self.bn2 = nn.BatchNorm1d(out_channels)
    self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,padding=padding2)
    self.relu = nn.ReLU(inplace=True)

    # projection residual
    if any([in_channels != out_channels, stride != 1]):
      self.residual = nn.Sequential(
          nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
          )
    # identity residual
    else:
      self.residual = nn.Sequential()

  def forward(self, x):
    out = self.relu(self.bn1(x))
    out = self.conv1(out)
    out = self.relu(self.bn2(out))
    out = self.conv2(out)
    out += self.residual(x)
    return out


class MethylCNNv1(nn.Module):
    def __init__(self, features: FeatureSet, sequence_length = 32, num_classes = 2, dropout_p = 0.1):
        super(MethylCNNv1, self).__init__()

        self.features = features
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        if self.features == FeatureSet.NUCLEOTIDES:
          self.in_channels = 4
        elif self.features == FeatureSet.KINETICS:
          self.in_channels = 4
        elif self.features == FeatureSet.ALL:
          self.in_channels = 8
        else:
          raise ValueError('Invalid feature set. See FeatureSet class.')

        self.extractor = nn.Sequential(ResBlock(self.in_channels, self.in_channels*2, kernel_size=7),             # (B, 8, 32) -> (B, 8, 32)
                                       ResBlock(self.in_channels*2, self.in_channels*2,kernel_size=5),            # (B, 16, 32) -> (B, 16, 32)
                                       ResBlock(self.in_channels*2, self.in_channels*2,kernel_size=5),            # (B, 16, 32) -> (B, 16, 32)

                                       ResBlock(self.in_channels*2, self.in_channels*4,kernel_size=5),            # (B, 16, 32) -> (B, 32, 32)
                                       ResBlock(self.in_channels*4, self.in_channels*4,kernel_size=3),            # (B, 32, 32) -> (B, 32, 32)
                                       ResBlock(self.in_channels*4, self.in_channels*4,kernel_size=3),            # (B, 32, 32) -> (B, 32 32)

                                       ResBlock(self.in_channels*4, self.in_channels*8,kernel_size=3, stride=2),  # (B, 32, 32) -> (B, 64, 16)
                                       ResBlock(self.in_channels*8, self.in_channels*8,kernel_size=3),            # (B, 64, 16) -> (B, 64, 16)
                                       ResBlock(self.in_channels*8, self.in_channels*8,kernel_size=3),            # (B, 64, 16) -> (B, 64, 16)

                                       ResBlock(self.in_channels*8, self.in_channels*16,kernel_size=3, stride=2), # (B, 64, 16) -> (B, 128, 8)
                                       ResBlock(self.in_channels*16, self.in_channels*16,kernel_size=3),          # (B, 128, 8) -> (B, 128, 8)
                                       ResBlock(self.in_channels*16, self.in_channels*16,kernel_size=3),          # (B, 128, 8) -> (B, 128, 8)
                                       )

        # calculate fc layer input with dummy passthrough
        self.fc_input_features = self._get_conv_output_size(sequence_length)

        # Linear layers
        self.fc1 = nn.Linear(in_features=self.fc_input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)


        # dropout
        self.dropout = nn.Dropout(p=dropout_p)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
      return self.extractor(x)

    def _get_conv_output_size(self, sequence_length: int) -> int:
        """
        Calculates the flattened output size of the convolutional layers
        by performing a forward pass on random data of the right shape.
        """
        dummy_input = torch.randn(1, self.in_channels, sequence_length)
        # calculate number of elements (numel) of final feature map
        output = self._extract_features(dummy_input)
        return output.numel()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch['seq']
        kinetics = batch['kinetics']

        # the input is a dictionary, so convert to a tensor
        if self.features == FeatureSet.NUCLEOTIDES:
          x = seq.to(self.fc1.weight.dtype) # -> [B, 4, L]
        elif self.features == FeatureSet.KINETICS:
          x = kinetics.to(self.fc1.weight.dtype) # -> [B, 4, L]
        elif self.features == FeatureSet.ALL:
          x = torch.cat([seq, kinetics], dim=1).to(self.fc1.weight.dtype) # -> [B, 8, L]

        x = self._extract_features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
    
MODEL_REGISTRY = {
    'MethylCNNv1': MethylCNNv1,
    # 'MethylCNNv2': MethylCNNv2, # (when new models are added, note here)
}

