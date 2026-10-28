"""Independent finite-volume WENO specification, using exact rational algebra.

Reconstruct p(x) from its averages on unit cells at prescribed integer offsets.
The face is x=1/2. The Jiang-Shu indicator is the sum, for derivative orders
1 through degree(p), of integrals of (p^(l)(x))^2 over the central cell [-1/2,1/2].
No ERF coefficients, ideal weights or smoothness expressions enter this module.
These are uniform-grid, cell-average, exact-arithmetic specifications; they do
not establish the convergence order of the nonlinear weights near critical points.
"""
from fractions import Fraction
from functools import cache
from math import factorial


def moment(center, degree):
    """Integral of x**degree over a unit cell centered at center."""
    return ((Fraction(center) + Fraction(1, 2)) ** (degree + 1)
            - (Fraction(center) - Fraction(1, 2)) ** (degree + 1)) / (degree + 1)


def solve(matrix, targets):
    """Exact, possibly overdetermined linear system with multiple RHS columns.

    Require a unique solution and consistency in every row; never fit a subset
    of the equations or silently use a least-squares approximation.
    """
    rows, columns, rhs = len(matrix), len(matrix[0]), len(targets[0])
    if len(targets) != rows or any(len(row) != columns for row in matrix) or any(len(row) != rhs for row in targets):
        raise ValueError('inconsistent reference linear-system dimensions')
    augmented = [[Fraction(value) for value in (*row, *target)]
                 for row, target in zip(matrix, targets)]
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if augmented[row][column]), None)
        if pivot is None:
            raise ValueError('reference reconstruction does not have a unique solution')
        augmented[pivot_row], augmented[pivot] = augmented[pivot], augmented[pivot_row]
        divisor = augmented[pivot_row][column]
        augmented[pivot_row] = [value / divisor for value in augmented[pivot_row]]
        for row in range(rows):
            if row != pivot_row:
                factor = augmented[row][column]
                augmented[row] = [value - factor * basis
                                  for value, basis in zip(augmented[row], augmented[pivot_row])]
        pivot_row += 1
    if any(any(row) for row in augmented[pivot_row:]):
        raise ValueError('inconsistent reference reconstruction constraints')
    return tuple(tuple(row[columns:]) for row in augmented[:columns])


@cache
def reconstruction(offsets):
    """Map cell averages to ascending monomial coefficients of p(x)."""
    if not offsets or len(offsets) != len(set(offsets)):
        raise ValueError('reference stencil must have distinct cells')
    size = len(offsets)
    return solve([[moment(center, degree) for degree in range(size)] for center in offsets],
                 [[int(row == column) for column in range(size)] for row in range(size)])


@cache
def face(offsets):
    coefficients = reconstruction(offsets)
    return tuple(sum((row[column] * Fraction(1, 2) ** degree
                      for degree, row in enumerate(coefficients)), Fraction(0))
                 for column in range(len(offsets)))


@cache
def smoothness(offsets):
    """Symmetric matrix Q such that the derivative integral is samples^T Q samples."""
    coefficients = reconstruction(offsets)
    size = len(offsets)
    matrix = [[Fraction(0) for _ in offsets] for _ in offsets]
    for derivative in range(1, size):
        for left in range(derivative, size):
            for right in range(derivative, size):
                factor = (Fraction(factorial(left), factorial(left - derivative))
                          * Fraction(factorial(right), factorial(right - derivative))
                          * moment(0, left + right - 2 * derivative))
                for row in range(size):
                    for column in range(size):
                        matrix[row][column] += factor * coefficients[left][row] * coefficients[right][column]
    return tuple(tuple(row) for row in matrix)


@cache
def ideal_weights(stencils):
    """Unique combination of substencil face values matching the full stencil.

    MZQ uses a corrected full-stencil candidate and must not use this ordinary
    WENO blend specification: its positive unit-sum ideal weights are not unique.
    """
    full = tuple(sorted(set().union(*stencils)))
    columns = [dict(zip(stencil, face(stencil))) for stencil in stencils]
    solution = solve([[column.get(cell, Fraction(0)) for column in columns] for cell in full],
                     [[value] for value in face(full)])
    return tuple(row[0] for row in solution)
