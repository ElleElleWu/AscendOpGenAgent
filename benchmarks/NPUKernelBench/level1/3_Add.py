import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs element-wise addition with broadcasting support.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        """
        return torch.add(x, y, alpha=alpha)
