from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from dataclasses import dataclass


@dataclass
class ResNetConfig:
    """
    Configuration for the ResNet architecture:
        - board_size: Size of the Go board (default is 9 for 9x9 Go).
        - in_channels: Number of input channels (2 input planes: [self, opponent] from the current player's perspective).
            - e.g., input tensor/shape: (B, 2, 9, 9) for batch size B, 2 channels, 9x9 board.
        - n_channels: Number of output channels for the convolutional layers (the trunk width: the number of feature maps inside the stem and every residual block, not the final outputs).
            - e.g., output tensor/shape after trunk: (B, 64, 9, 9) if out_channels=64.
        - num_blocks: Number of residual blocks in the network.
        - policy_hidden: First 1x1 in policy head
        - value_hidden: Number of hidden units in the value head's fully connected layer.
        - policy_pass_from_global: Whether to keep the PASS logit from pooled features.
    """
    board_size: int = 9
    in_channels: int = 2                  # [self, opponent]
    n_channels: int = 64                  # trunk width
    num_blocks: int = 8                   # number of residual blocks
    policy_hidden: int = 32               # first 1x1 in policy head
    value_channels: int = 32              # channels in value head conv
    value_hidden: int = 128               # first 1x1 in value head
    policy_pass_from_global: bool = True


class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class ResidualBlock(nn.Module):
    """
    A pre-activation residual block:
        - BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU -> Conv2d -> Add input (skip connection)
        - In each convolution layer:
            - kernel_size=3, padding=1 to preserve spatial dimensions.
            - NOTE: keep stride=1 as default to preserve the 9x9 spatial resolution so each feature stays aligned with a board intersection. \
                    Downsampling (stride>1 or pooling) would shrink the grid and lose locality.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
    
        if in_channels != out_channels:
            self.proj = ShortcutProjection(in_channels, out_channels)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return self.proj(x) + out  # Skip connection


class ResNet(nn.Module):
    """
    ResNet architecture with policy and value heads for 9x9 Go:
        - Initial convolutional stem.
        - Stack of residual blocks.
        - Policy head: outputs move probabilities for each board position + optional PASS logit.
        - Value head: outputs a scalar value estimating the win probability.
    
    Inputs X: [B, in_channels, N, N]
        - B: batch size
        - in_channels: number of input channels (2 for self and opponent)
        - N: board size (9 for 9x9 Go)
    Outputs:
        - Policy logits: [B, N*N + 1] (avaiable actions for 9x9 board game: 81 positions + 1 PASS move)
        - Value: [B, 1] (one scalar evaluation per position in the batch)
    """
    def __init__(self, cfg: ResNetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Initial convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.n_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.n_channels),
            nn.ReLU(inplace=True)
        )
        
        # Trunk: stack of residual blocks, constant width
        blocks = []
        for _ in range(cfg.num_blocks):
            blocks.append(ResidualBlock(cfg.n_channels, cfg.n_channels))
        self.trunk = nn.Sequential(*blocks)
        
        # Policy head (board logits)
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.n_channels, cfg.policy_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.policy_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.policy_hidden, 1, kernel_size=1, bias=True)  # Output: [B, 1, N, N]
        )
        
        # Separate PASS logit from pooled features
        if cfg.policy_pass_from_global:
            self.pass_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),        # Global average pooling: [B, C, 1, 1], C = n_channels
                nn.Flatten(),                   # [B, C]
                nn.Linear(cfg.n_channels, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)                # Output: [B, 1]
            )
        else:
            self.pass_bias = nn.Parameter(torch.zeros(1))
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.n_channels, cfg.value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.value_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.value_channels, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1)    # Output: [B, 1]
        )
        
        # Initialize blocks as near-identity for stability
        self._init_identityish()
    
    def _init_identityish(self) -> None:
        """
        Zero-initialize the last BN's gamma in each residual block so that
        each block starts as (approximately) an identity mapping.
        """
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if hasattr(m.bn2, "weight") and m.bn2.weight is not None:
                    nn.init.zeros_(m.bn2.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input tensor x: [B, in_channels, N, N]
        Outputs:
            - Policy logits: [B, N*N + 1]
            - Value: [B, 1]
        """
        B, _, N, _ = x.shape
        
        h = self.stem(x)          # [B, n_channels, N, N]
        h = self.trunk(h)         # [B, n_channels, N, N]
        
        # Policy head
        policy_board = self.policy_head(h).view(B, -1)  # [B, N*N]
        
        # PASS logit
        if self.cfg.policy_pass_from_global:
            policy_pass = self.pass_head(h)  # [B, 1]
        else:
            policy_pass = self.pass_bias.expand(B, 1)  # [B, 1]
        
        policy = torch.cat([policy_board, policy_pass], dim=1)  # [B, N*N + 1] = [B, 82]
        
        # Value head
        value = torch.tanh(self.value_head(h))  # [B, 1]
        
        return policy, value


if __name__ == "__main__":
    # simple test
    cfg = ResNetConfig()
    net = ResNet(cfg)
    x = torch.randn(4, cfg.in_channels, cfg.board_size, cfg.board_size)
    p, v = net(x)
    assert p.shape == (4, cfg.board_size * cfg.board_size + 1), p.shape
    assert v.shape == (4, 1), v.shape
    print("OK:", p.shape, v.shape)