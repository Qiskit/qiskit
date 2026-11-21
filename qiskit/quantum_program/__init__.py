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

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   DataTree
   DType
   TensorType
   bounded
"""

from qiskit._accelerate.quantum_program import (
    DataTree,
    DType,
    TensorType,
    bounded,
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
    "TensorType",
    "bit",
    "bounded",
    "c64",
    "c128",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
]
