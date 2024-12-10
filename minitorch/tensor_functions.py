"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend


if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)

        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply element-wise negation to the input tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Apply element-wise negation to the gradient of the output."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Apply element-wise inverse to the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Apply element-wise inverse to the gradient of the output."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Add two tensors element-wise"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Subtract two tensors element-wise"""
        return t1.f.add_zip(t1, -t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        return grad_output, -grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        ctx.save_for_backward(a, dim)
        if dim._tensor.size > 1:
            for i in range(dim._tensor.size):
                a = a.f.mul_reduce(a, i)
            a = tensor([a._tensor._storage[0]])
        else:
            a = a.f.mul_reduce(a, int(dim._tensor._storage[0]))
        return a

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Return the gradients of the output with respect to the inputs"""
        a, dim = ctx.saved_values
        # Gradient of all is non-trivial; assuming binary inputs, gradient is zero
        grad_a = a.f.zeros_like(a)
        return grad_a, None


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Multiply two tensors element-wise"""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        a, b = ctx.saved_values
        grad_a = grad_output.f.mul_zip(grad_output, b)
        grad_b = grad_output.f.mul_zip(grad_output, a)
        return grad_a, grad_b


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Apply element-wise sigmoid to the input tensor."""
        out = a.f.sigmoid_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the output with respect to the input"""
        (sigmoid_a,) = ctx.saved_values
        one = minitorch.Tensor.make(
            [1.0] * sigmoid_a._tensor.size, sigmoid_a.shape, backend=sigmoid_a.backend
        )
        one_minus_sigmoid = one.f.add_zip(one, -sigmoid_a)
        sigmoid_derivative = sigmoid_a.f.mul_zip(sigmoid_a, one_minus_sigmoid)
        grad_input = grad_output.f.mul_zip(grad_output, sigmoid_derivative)
        return grad_input


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Apply element-wise ReLU to the input tensor."""
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the output with respect to the input"""
        (a,) = ctx.saved_values
        grad = grad_output.f.relu_back_zip(a, grad_output)
        return grad


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Apply element-wise logarithm to the input tensor."""
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the output with respect to the input"""
        (a,) = ctx.saved_values
        grad_input = grad_output.f.mul_zip(grad_output, Inv.apply(a))
        return grad_input


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Apply element-wise exponential to the input tensor."""
        out = a.f.exp_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the output with respect to the input"""
        (exp_out,) = ctx.saved_values
        grad = exp_out.f.mul_zip(grad_output, exp_out)
        return grad


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Sum function.

        Args:
        ----
            ctx (Context): Context to save information for backward.
            a (Tensor): Input tensor to be summed.
            dim (Tensor): Dimensions along which to sum.

        Returns:
        -------
            Tensor: Result of the summation.

        """
        ctx.save_for_backward(a, dim)
        if dim._tensor.size > 1:
            for i in range(dim._tensor.size):
                a = a.f.add_reduce(a, i)
            a = tensor([a._tensor._storage[0]])
        else:
            a = a.f.add_reduce(a, int(dim._tensor._storage[0]))
        return a

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Sum function.

        Args:
        ----
            ctx (Context): Context containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to the inputs (a and dim).

        """
        a, dim = ctx.saved_values
        grad_a = a.expand(grad_output)
        grad_dim = grad_output.zeros(dim.shape)
        return grad_a, grad_dim


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compare two tensors element-wise"""
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        zero_a = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[0]._tensor.size,
            ctx.saved_values[0].shape,
            backend=ctx.saved_values[0].backend,
        )
        zero_b = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[1]._tensor.size,
            ctx.saved_values[1].shape,
            backend=ctx.saved_values[1].backend,
        )
        return zero_a, zero_b


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compare two tensors element-wise"""
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        zero_a = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[0]._tensor.size,
            ctx.saved_values[0].shape,
            backend=ctx.saved_values[0].backend,
        )
        zero_b = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[1]._tensor.size,
            ctx.saved_values[1].shape,
            backend=ctx.saved_values[1].backend,
        )
        return zero_a, zero_b


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compare two tensors element-wise"""
        ctx.save_for_backward(a, b)
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradients of the output with respect to the inputs"""
        zero_a = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[0]._tensor.size,
            ctx.saved_values[0].shape,
            backend=ctx.saved_values[0].backend,
        )
        zero_b = minitorch.Tensor.make(
            [0.0] * ctx.saved_values[1]._tensor.size,
            ctx.saved_values[1].shape,
            backend=ctx.saved_values[1].backend,
        )
        return zero_a, zero_b


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the input tensor"""
        ctx.save_for_backward(order)
        # assert a._tensor.is_contiguous(), "Must be contiguous to permute"

        # Convert order tensor to list of integers
        order2 = [int(order[i]) for i in range(order.size)]

        # Validate permutation order length
        if len(order2) != len(a.shape):
            raise ValueError(
                f"Permutation order length {len(order2)} does not match tensor dimensions {len(a.shape)}."
            )

        # Validate permutation indices
        if sorted(order2) != list(range(len(a.shape))):
            raise ValueError(
                f"Invalid permutation order {order2}. It must be a permutation of {list(range(len(a.shape)))}."
            )

        # Permute shape and strides
        permuted_shape = tuple(a.shape[i] for i in order2)
        permuted_strides = tuple(a._tensor.strides[i] for i in order2)

        # Create permuted tensor
        return minitorch.Tensor.make(
            a._tensor._storage, permuted_shape, permuted_strides, backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the output with respect to the input"""
        (order,) = ctx.saved_tensors
        inverse_order_storage = [0] * order._tensor.size
        for i in range(order._tensor.size):
            index = int(order._tensor._storage[i])  # Ensure the index is an integer
            inverse_order_storage[index] = i
        inverse_order = tensor(inverse_order_storage)
        return (
            Permute.apply(grad_output, inverse_order),
            zeros(inverse_order.shape),
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """View the tensor with the given shape"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference gradient for a given function and tensor arguments."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()

    random.seed(10)
    out = f(*vals)

    out.sum().backward()

    err_msg = """
    Gradient check error for function %s.

    Input %s

    Received derivative %f for argument %d and index %s,
    but was expecting derivative %f from central difference.

    """
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
