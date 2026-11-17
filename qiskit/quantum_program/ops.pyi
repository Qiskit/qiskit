# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The classes here are declared by hand because they come from the compiled extension module,
# which has no signatures of its own. Each member's summary line is repeated so that a hover
# keeps showing one; the full documentation lives on the Rust type. Nothing may be declared here
# that the class does not have, which test_quantum_program.py checks. A class declares only its own
# members and inherits the rest, and every type it names is imported absolutely, so that the test
# reads no import here as an export.

from typing import TYPE_CHECKING

from qiskit.quantum_program import DType, TensorType, bounded

__all__ = [
    "Add",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "BroadcastTo",
    "Cast",
    "Constant",
    "Divide",
    "Mean",
    "Multiply",
    "Parity",
    "Power",
    "ProgramOp",
    "Remainder",
    "Std",
    "Subtract",
    "Variance",
]

class ProgramOp:
    """One of the operations an instruction can perform."""

    @property
    def full_name(self) -> str:
        """The type name, qualified by its namespace."""

    def output_types(self, operands: Sequence[TensorType], /) -> list[TensorType]:
        """The types this operation produces from operands of type ``operands``."""

class Add(ProgramOp):
    """Add two tensors elementwise."""

    def __init__(self) -> None: ...

class Subtract(ProgramOp):
    """Subtract the second tensor from the first, elementwise."""

    def __init__(self) -> None: ...

class Multiply(ProgramOp):
    """Multiply two tensors elementwise."""

    def __init__(self) -> None: ...

class Divide(ProgramOp):
    """Divide the first tensor by the second, elementwise."""

    def __init__(self) -> None: ...

class Remainder(ProgramOp):
    """Remainder of the first tensor divided by the second, elementwise."""

    def __init__(self) -> None: ...

class Power(ProgramOp):
    """Raise the first tensor to the power of the second, elementwise."""

    def __init__(self) -> None: ...

class BitwiseAnd(ProgramOp):
    """Bitwise AND of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseOr(ProgramOp):
    """Bitwise OR of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseXor(ProgramOp):
    """Bitwise XOR of two bit-valued tensors, elementwise."""

    def __init__(self) -> None: ...

class BitwiseNot(ProgramOp):
    """Bitwise NOT of a bit-valued tensor, elementwise."""

    def __init__(self) -> None: ...

class Parity(ProgramOp):
    """XOR-reduce the bits of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /) -> None: ...

class Mean(ProgramOp):
    """Average a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /) -> None: ...

class Variance(ProgramOp):
    """Variance of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /, ddof: float = 0.0) -> None: ...

class Std(ProgramOp):
    """Standard deviation of a tensor along one axis, removing that axis."""

    def __init__(self, axis: int, /, ddof: float = 0.0) -> None: ...

class Cast(ProgramOp):
    """Reinterpret a tensor as another dtype, keeping its shape."""

    def __init__(self, target: DType, /) -> None: ...

class BroadcastTo(ProgramOp):
    """Broadcast a tensor to a shape, aligning its axes with the trailing axes of that shape."""

    def __init__(self, target: Sequence[int | bounded], /) -> None: ...

class Constant(ProgramOp):
    """Supply a tensor the program holds, rather than one given at call time."""

    def __init__(self, value: ArrayLike, /) -> None: ...

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
