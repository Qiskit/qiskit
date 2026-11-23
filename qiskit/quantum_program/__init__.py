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

"""
=================================================
Quantum programs (:mod:`qiskit.quantum_program`)
=================================================

.. currentmodule:: qiskit.quantum_program

A quantum program describes a hybrid quantum-classical computation as typed tensor values
produced and consumed by instructions. It is inert data: it describes a computation without
performing one.

A program's inputs and outputs travel in a data tree, arranged by the structures the program
declares, which is where all naming lives.

Writing a program
=================

:func:`qp_input` declares a typed input and returns a :class:`Tracer`, which the arithmetic
operators, the bitwise operators and the reductions build on. Each operation reports the type it
produces as it is written, so a dtype or shape mistake is raised at the line that made it::

    from qiskit.quantum_program import build, f64, qp_input

    x = qp_input("x", f64[3])
    y = (x * x).mean(axis=0)
    y.type                        # TensorType(F64[])

:func:`build` names the outputs, with whatever nesting they should come back in, and gives the
program. Calling it takes one keyword argument per declared input::

    program = build({"z": y})
    program.output_types()        # DataTree([z: TensorType(F64[])])
    program(x=[1.0, 2.0, 3.0])    # DataTree([z: array(4.66666667)])

Only the expressions the declared outputs reach are built, so a value used twice becomes one
instruction and a value nothing uses costs nothing.

Operations
==========

Each operation an instruction can perform has a class in :mod:`qiskit.quantum_program.ops`, which
both builds the operation and reads back the payload it was built with. The functions here are what
writes a program, and they apply those operations for you, so a class is what you reach for to read
an operation back::

    from qiskit.quantum_program import ops

    ops.Mean(0).full_name    # 'qiskit.mean'

Dtypes
======

A dtype is a member of :class:`DType`. Each has a lower-case alias, and indexing one
with a shape gives the :class:`TensorType` of a value::

    f64[3]                        # a vector of three 64-bit floats
    bit[4000, 2]                  # a 4000x2 array of bits
    f64[()]                       # a scalar
    bit[1024, bounded(64)]        # an axis whose size is known only at run time

The current aliases are ``bit``, ``u8``, ``u16``, ``u32``, ``u64``, ``i8``, ``i16``, ``i32``,
``i64``, ``f32``, ``f64``, ``c64`` and ``c128``.

Functions
=========

.. autosummary::
   :toctree: ../stubs/

   qp_input
   constant
   build
   add
   subtract
   multiply
   divide
   remainder
   power
   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_not
   mean
   var
   std
   parity
   cast
   broadcast_to

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   DataTree
   DType
   QuantumProgram
   TensorType
   Tracer
   Value
   bounded

Submodules
==========

.. autosummary::
   :toctree: ../stubs/

   ops
"""

from qiskit._accelerate.quantum_program import (
    DataTree,
    DType,
    QuantumProgram,
    TensorType,
    Value,
    bounded,
)

from . import ops
from ._tracer import (
    Tracer,
    add,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    build,
    cast,
    constant,
    divide,
    mean,
    multiply,
    parity,
    power,
    qp_input,
    remainder,
    std,
    subtract,
    var,
)

bit = DType.Bit
u8 = DType.U8
u16 = DType.U16
u32 = DType.U32
u64 = DType.U64
i8 = DType.I8
i16 = DType.I16
i32 = DType.I32
i64 = DType.I64
f32 = DType.F32
f64 = DType.F64
c64 = DType.C64
c128 = DType.C128

__all__ = [
    "DType",
    "DataTree",
    "QuantumProgram",
    "TensorType",
    "Tracer",
    "Value",
    "add",
    "bit",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bounded",
    "broadcast_to",
    "build",
    "c64",
    "c128",
    "cast",
    "constant",
    "divide",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "mean",
    "multiply",
    "ops",
    "parity",
    "power",
    "qp_input",
    "remainder",
    "std",
    "subtract",
    "u8",
    "u16",
    "u32",
    "u64",
    "var",
]
