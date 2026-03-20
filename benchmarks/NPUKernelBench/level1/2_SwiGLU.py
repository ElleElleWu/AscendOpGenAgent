import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a SwiGLU activation.
    SwiGLU(x, dim) = Swish(a) * b, where a and b are chunks of x along dim.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Applies SwiGLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor where the size of dim must be even.
            dim (int, optional): The dimension along which to chunk the tensor.

        Returns:
            torch.Tensor: Output tensor with SwiGLU applied, shape is same as x except
                          dim is halved.
        """
        a, b = torch.chunk(x, 2, dim=dim)
        return torch.nn.functional.silu(a) * b
