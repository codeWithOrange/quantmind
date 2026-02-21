from __future__ import annotations
from typing import Iterable, Iterator
import math


class Vector:
    """
    Immutable mathematical vector.
    """

    __slots__ = ("_components",)

    def __init__(self, components: Iterable[float]):
        comps = tuple(float(x) for x in components)
        if not comps:
            raise ValueError("Vector cannot be empty.")
        self._components = comps

    # --------------------------
    # Core Methods
    # --------------------------

    def __repr__(self):
        return f"Vector({self._components})"

    def __len__(self):
        return len(self._components)

    def __iter__(self) -> Iterator[float]:
        return iter(self._components)

    def __getitem__(self, index: int):
        return self._components[index]

    def __eq__(self, other: Vector):
        if not isinstance(other, Vector):
            return False
        return all(math.isclose(a, b) for a, b in zip(self, other))

    def __hash__(self):
        return hash(self._components)

    # --------------------------
    # Arithmetic Operations
    # --------------------------

    def __add__(self, other: Vector) -> Vector:
        self._check_dim(other)
        return Vector(a + b for a, b in zip(self, other))

    def __sub__(self, other: Vector) -> Vector:
        self._check_dim(other)
        return Vector(a - b for a, b in zip(self, other))

    def __mul__(self, scalar: float) -> Vector:
        return Vector(a * scalar for a in self)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vector:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector(a / scalar for a in self)

    def __neg__(self) -> Vector:
        return Vector(-a for a in self)

    # --------------------------
    # Dot Product using @
    # --------------------------

    def __matmul__(self, other: Vector) -> float:
        return self.dot(other)

    def dot(self, other: Vector) -> float:
        self._check_dim(other)
        return sum(a * b for a, b in zip(self, other))

    # --------------------------
    # Magnitude & Normalize
    # --------------------------

    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> Vector:
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector.")
        return self / mag

    # --------------------------
    # Advanced Operations
    # --------------------------

    def distance(self, other: Vector) -> float:
        return (self - other).magnitude()

    def angle(self, other: Vector) -> float:
        self._check_dim(other)
        denom = self.magnitude() * other.magnitude()
        if denom == 0:
            raise ValueError("Cannot compute angle with zero vector.")
        return math.acos((self @ other) / denom)

    def hadamard(self, other: Vector) -> Vector:
        self._check_dim(other)
        return Vector(a * b for a, b in zip(self, other))

    def projection(self, other: Vector) -> Vector:
        scalar = (self @ other) / (other @ other)
        return scalar * other

    # --------------------------
    # Internal Utility
    # --------------------------

    def _check_dim(self, other: Vector):
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimension.")
