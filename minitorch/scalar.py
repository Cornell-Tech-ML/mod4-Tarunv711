from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
    EQ,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()

    # def __post_init__(self):
    #     global _var_count
    #     _var_count += 1
    #     object.__setattr__(self, "unique_id", _var_count)
    #     object.__setattr__(self, "name", str(self.unique_id))
    #     object.__setattr__(self, "data", float(self.data))

    # def __repr__(self) -> str:
    #     return f"Scalar({self.data})"

    # # ## Task 1.2 and 1.4
    # def __lt__(self, b: ScalarLike) -> Scalar:
    #     return LT.apply(self, b)

    # def __gt__(self, b: ScalarLike) -> Scalar:
    #     return LT.apply(b, self)

    # def __sub__(self, b: ScalarLike) -> Scalar:
    #     return Add.apply(self, Neg.apply(b))

    # def __neg__(self) -> Scalar:
    #     return Neg.apply(self)

    # def __add__(self, b: ScalarLike) -> Scalar:
    #     return Add.apply(self, b)

    # def log(self) -> Scalar:
    #     """Apply the logarithm function to the current scalar."""
    #     return Log.apply(self)

    # def exp(self) -> Scalar:
    #     """Apply the exponential function to the current scalar."""
    #     return Exp.apply(self)

    # def sigmoid(self) -> Scalar:
    #     """Apply the sigmoid function to the current scalar."""
    #     return Sigmoid.apply(self)

    # def relu(self) -> Scalar:
    #     """Apply the ReLU function to the current scalar."""
    #     return ReLU.apply(self)


# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: float = 0.0
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __hash__(self) -> int:
        return hash(self.data)

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __add__(self, other: ScalarLike) -> Scalar:
        return Add.apply(self, other)

    def __radd__(self, other: ScalarLike) -> Scalar:
        return Add.apply(other, self)

    def __sub__(self, other: ScalarLike) -> Scalar:
        return self + (-other)

    def __rsub__(self, other: ScalarLike) -> Scalar:
        return other + (-self)

    def __mul__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(self, other)

    def __rmul__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(other, self)

    def __truediv__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(other))

    def __rtruediv__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(Inv.apply(self), other)

    def __eq__(self, other: ScalarLike) -> Scalar:
        return EQ.apply(self, other)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __lt__(self, other: ScalarLike) -> Scalar:
        return LT.apply(self, other)

    def __gt__(self, other: ScalarLike) -> Scalar:
        return LT.apply(other, self)

    def log(self) -> Scalar:
        """Apply the logarithm function to the current scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Apply the exponential function to the current scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Apply the sigmoid function to the current scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Apply the ReLU function to the current scalar."""
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0

        if type(x) is tuple:
            x = x[0]

        if type(x) is Scalar:
            x = x.data

        if not isinstance(x, (int, float)):
            raise TypeError(f"Unsupported type for derivative: {type(x)}")

        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable was created by an operation."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to get the derivatives of the input variables.

        Args:
        ----
            d_output: The derivative of the output variable.

        Returns:
        -------
            Iterable of tuples with the input variables and their derivatives.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # TODO: Implement for Task 1.3.
        # Get the backward function and apply it to the context and d_output
        grads = h.last_fn._backward(h.ctx, d_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        # Pair the gradients with their corresponding input variables
        # Filter out constants (variables without history)
        return [
            (inp, grad) for inp, grad in zip(h.inputs, grads) if not inp.is_constant()
        ]

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Check the derivatives of the output variable using central difference.

    Args:
    ----
        f: The function to differentiate.
        scalars: The input scalars.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
