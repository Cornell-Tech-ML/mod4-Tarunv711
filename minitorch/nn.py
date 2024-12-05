from typing import Tuple

from .tensor import Tensor
from .tensor_functions import Function,rand, tensor
from .fast_ops import FastOps
from . import operators
from .autodiff import Context



# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off
fast_max = FastOps.reduce(operators.max, -float("inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    # Reshape input to batch x channel x new_height x kernel_height x new_width x kernel_width
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Transpose and reshape to get batch x channel x new_height x new_width x (kernel_height * kernel_width)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width
def argmax(input: Tensor, dim: int = -1) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply argmax

    Returns:
    -------
        Tensor of size batch x channel x height x width x 1

    """
    # Create a tensor of zeros with the same shape as the input
    # Set the dimension to apply argmax to the maximum value
    max_tensor = fast_max(input, dim)
    return max_tensor == input

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Validate input dimensions
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    # Reshape input into tiles
    tiled_input, new_height, new_width = tile(input, kernel)

    # Take mean over the last dimension (kernel_height * kernel_width)
    pooled = tiled_input.mean(dim=4)

    # Add this line to ensure correct shape
    return pooled.contiguous().view(batch, channel, new_height, new_width)

class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation.

        Args:
        ----
            ctx : Context
            a : input tensor
            dim : dimension to reduce

        Returns:
        -------
            Tensor: maximum values

        """
        # Save input for backward pass
        ctx.save_for_backward(a, dim)
        # Get the maximum value along dimension
        return fast_max(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        t1, dim = ctx.saved_values
        max_mask = argmax(t1, int(dim.item()))
        # The issue was here - we don't need to divide by the number of max values
        return max_mask * grad_output, 0.0




def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max reduction.

    Args:
    ----
        input: input tensor
        dim: dimension to reduce

    Returns:
    -------
        Tensor of maximum values

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the input tensor along the specified dimension.

    Args:
    ----
        input : Tensor
            Input tensor.
        dim : int
            Dimension along which to compute softmax.

    Returns:
    -------
        Tensor
            Softmax of the input tensor.

    """
    exps = input.exp()
    sum_exps = exps.sum(dim)
    # Reshape sum_exps to allow broadcasting
    shape = list(exps.shape)
    shape[dim] = 1
    sum_exps = sum_exps.contiguous().view(*shape)
    return exps / sum_exps


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax of the input tensor along the specified dimension.

    Args:
    ----
        input : Tensor
            Input tensor.
        dim : int
            Dimension along which to compute log softmax.

    Returns:
    -------
        Tensor
            Log softmax of the input tensor.

    """
    softmax_tensor = softmax(input, dim)
    return softmax_tensor.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input : Tensor
            Input tensor with shape (batch, channel, height, width).
        kernel : Tuple[int, int]
            Height and width of the pooling kernel.

    Returns:
    -------
        Tensor
            Pooled tensor with shape (batch, channel, new_height, new_width).

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Validate input dimensions
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Use tile function to reshape input into tiles
    tiled_input, new_height, new_width = tile(input, kernel)

    # Take max over the kernel dimension (last dimension)
    pooled = max(tiled_input, dim=4)

    # Reshape to final output dimensions
    return pooled.contiguous().view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Randomly zeroes some of the elements of the input tensor with probability p.

    During training, each element is zeroed out with probability `p` and scaled by `1/(1-p)`.
    During evaluation, the input is returned unchanged.

    Args:
    ----
        input : Tensor
            Input tensor.
        p : float
            Probability of an element to be zeroed. Must be between 0 and 1.
        ignore : bool
            If True, returns input unchanged. Default False.

    Returns:
    -------
        Tensor
            Tensor with elements randomly zeroed.

    """
    if ignore:
        return input

    if not 0 <= p <= 1:
        raise ValueError("Dropout probability must be in the range [0, 1).")

    if p == 0.0:
        return input

    if p == 1.0:
        return input * 0.0

    # Create a mask with the same shape as input
    random = rand(input.shape)
    prob_tensor = tensor([p])  # Create a tensor with single value
    mask = random.f.lt_zip(random, prob_tensor)
    # Scale the input to maintain the expected value
    return input * (tensor(1.0) - mask) / (tensor(1.0) - p)
