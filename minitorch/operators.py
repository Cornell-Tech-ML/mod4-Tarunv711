"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two floating-point numbers.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function for a floating-point number.

    Args:
    ----
        x: A floating-point number.

    Returns:
    -------
        The input number x unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floating-point numbers.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def sub(x: float, y: float) -> float:
    """Sub two floating-point numbers.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        The sub of x and y.

    """
    return x - y


def neg(x: float) -> float:
    """Negate a floating-point number.

    Args:
    ----
        x: A floating-point number.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if one floating-point number is less than another.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if two floating-point numbers are equal.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Find the maximum of two floating-point numbers.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two floating-point numbers are close to each other.

    Args:
    ----
        x: First floating-point number.
        y: Second floating-point number.

    Returns:
    -------
        True if the absolute difference between x and y is less than 1e-2, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function of a floating-point number.

    Args:
    ----
        x: A floating-point number.

    Returns:
    -------
        The sigmoid of x, calculated as 1 / (1 + e^(-x)) if x >= 0, else e^x / (1 + e^x).

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the rectified linear unit (ReLU) of a floating-point number.

    Args:
    ----
        x: A floating-point number.

    Returns:
    -------
        The ReLU of x, which is max(0, x).

    """
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm of a floating-point number.

    Args:
    ----
        x: A positive floating-point number.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential of a floating-point number.

    Args:
    ----
        x: A floating-point number.

    Returns:
    -------
        e raised to the power of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm function.

    Args:
    ----
        x: The input value at which the gradient is evaluated.
        d: The upstream gradient.

    Returns:
    -------
        The gradient of the natural logarithm function at x.

    """
    return d / x


def inv(x: float) -> float:
    """Compute the multiplicative inverse of a floating-point number.

    Args:
    ----
        x: A non-zero floating-point number.

    Returns:
    -------
        The multiplicative inverse of x (1/x).

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the multiplicative inverse function.

    Args:
    ----
        x: The input value at which the gradient is evaluated.
        d: The upstream gradient.

    Returns:
    -------
        The gradient of the multiplicative inverse function at x.

    """
    return -(d / (x * x))


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the rectified linear unit (ReLU) function.

    Args:
    ----
        x: The input value at which the gradient is evaluated.
        d: The upstream gradient.

    Returns:
    -------
        The gradient of the ReLU function at x.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element in a list.

    Args:
    ----
        fn: A function that takes a float and returns a float.
        ls: A list of floats.

    Returns:
    -------
        A new list with the function applied to each element.

    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two lists.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        ls1: The first list of floats.
        ls2: The second list of floats.

    Returns:
    -------
        A new list with the function applied to each pair of elements.

    """
    it1, it2 = iter(ls1), iter(ls2)
    result = []
    while True:
        try:
            x, y = next(it1), next(it2)
            result.append(fn(x, y))
        except StopIteration:
            break
    return result


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Reduce a list to a  gle value by repeatedly applying a function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        ls: A list of floats.

    Returns:
    -------
        The final accumulated value.

    """
    it = iter(ls)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("Cannot reduce an empty iterable")
    for x in it:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        A new list with all elements negated.

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise.

    Args:
    ----
        ls1: The first list of floats.
        ls2: The second list of floats.

    Returns:
    -------
        A new list with elements added pairwise.

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Calculate the sum of all elements in a list.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, ls) if ls else 1.0
