import torch
import torch.nn as nn
import math

# 1. Initialize the registry
_SUBNET_REGISTRY = {}


def register_subnet(name):
    """Decorator to register a subnet class."""

    def decorator(cls):
        _SUBNET_REGISTRY[name] = cls
        return cls

    return decorator


def build_subnet(name, **kwargs):
    """Factory function to build a subnet by name string."""
    if name is None:
        return None
    if name not in _SUBNET_REGISTRY:
        raise ValueError(f"Subnet '{name}' is not found in registry. Available: {list(_SUBNET_REGISTRY.keys())}")
    return _SUBNET_REGISTRY[name](**kwargs)


# 2. Define Subnet Architectures

@register_subnet("SimpleAlphaBeta")
class SimpleAlphaBetaHead(nn.Module):
    """
    A lightweight head to predict alpha and beta for TAL.
    Structure: Conv(3x3) -> SiLU -> Conv(1x1)
    """

    def __init__(self, in_channels, hidden_channels=64, default_alpha=0.5, default_beta=6.0):
        super().__init__()
        self.default_alpha = default_alpha
        self.default_beta = default_beta

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, padding=0, bias=True)
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize the last layer bias to 0, weight to small values.
        # This ensures that at the beginning of training:
        # output ~= 0 -> tanh(0) = 0 -> exp(0) = 1
        # So alpha ~= default_alpha, beta ~= default_beta
        last_conv = self.net[-1]
        nn.init.constant_(last_conv.bias, 0)
        nn.init.normal_(last_conv.weight, mean=0, std=0.01)

    def forward(self, x):
        # x: [B, C, H, W]
        # out: [B, 2, H, W]
        out = self.net(x)
        delta_alpha, delta_beta = out.chunk(2, dim=1)

        # Mathematical constraints for stability:
        # 1. Use tanh to bound the fluctuation range.
        # 2. Use exp to ensure positivity.
        # 3. Scaling factors (3.0 and 2.0) control the dynamic range.

        # Alpha range: default * [exp(-3), exp(3)] ~= default * [0.05, 20.0]
        alpha = self.default_alpha * torch.exp(3.0 * torch.tanh(delta_alpha))

        # Beta range: default * [exp(-2), exp(2)] ~= default * [0.135, 7.38]
        # Tighter constraint for beta to prevent gradient explosion on tiny objects
        beta = self.default_beta * torch.exp(2.0 * torch.tanh(delta_beta))

        return torch.cat([alpha, beta], dim=1)


@register_subnet("BottleneckAlphaBeta")
class BottleneckAlphaBetaHead(nn.Module):
    """
    Uses a bottleneck structure for efficiency if input channels are large.
    """

    def __init__(self, in_channels, default_alpha=0.5, default_beta=6.0):
        super().__init__()
        self.default_alpha = default_alpha
        self.default_beta = default_beta
        reduced_dim = max(16, in_channels // 4)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False),  # Reduce
            nn.BatchNorm2d(reduced_dim),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=1, groups=reduced_dim, bias=False),  # DW
            nn.Conv2d(reduced_dim, 2, kernel_size=1, bias=True)  # Linear Proj
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.net[-1].bias, 0)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        out = self.net(x)
        delta_alpha, delta_beta = out.chunk(2, dim=1)
        alpha = self.default_alpha * torch.exp(3.0 * torch.tanh(delta_alpha))
        beta = self.default_beta * torch.exp(2.0 * torch.tanh(delta_beta))
        return torch.cat([alpha, beta], dim=1)

