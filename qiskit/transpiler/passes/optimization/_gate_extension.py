# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Dynamically extend Gate classes with functions required for the Hoare
optimizer, namely triviality-conditions and post-conditions.
If `_trivial_if` returns `True` and the qubit is in a classical state
then the gate is trivial.
If a gate has no `_trivial_if`, then is assumed to be non-trivial.
If a gate has no `_postconditions`, then is assumed to have unknown post-conditions.
"""
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, CYGate, CZGate
from qiskit.circuit.library.standard_gates import TGate, TdgGate, SGate, SdgGate, RZGate, U1Gate
from qiskit.circuit.library.standard_gates import SwapGate, CSwapGate, CRZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals

if _optionals.HAS_Z3:
    import z3  # pylint: disable=import-error

    # FLIP GATES #
    # XGate
    XGate._postconditions = lambda self, x1, y1: y1 == z3.Not(x1)  # type: ignore[attr-defined]
    CXGate._postconditions = lambda self, x1, y1: y1 == z3.Not(x1)  # type: ignore[attr-defined]
    CCXGate._postconditions = lambda self, x1, y1: y1 == z3.Not(x1)  # type: ignore[attr-defined]

    # YGate
    YGate._postconditions = lambda self, x1, y1: y1 == z3.Not(x1)  # type: ignore[attr-defined]
    CYGate._postconditions = lambda self, x1, y1: y1 == z3.Not(x1)  # type: ignore[attr-defined]

    # PHASE GATES #
    # IdGate
    IGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    # ZGate
    ZGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    ZGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    CZGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    CZGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    # SGate
    SGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    SGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    SdgGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    SdgGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    # TGate
    TGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    TGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    TdgGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    TdgGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    # RzGate
    RZGate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    RZGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    CRZGate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    #  U1Gate
    U1Gate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    U1Gate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    CU1Gate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    CU1Gate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]
    MCU1Gate._trivial_if = lambda self, x1: True  # type: ignore[attr-defined]
    MCU1Gate._postconditions = lambda self, x1, y1: y1 == x1  # type: ignore[attr-defined]

    # MULTI-QUBIT GATES #
    # SwapGate
    SwapGate._trivial_if = lambda self, x1, x2: x1 == x2  # type: ignore[attr-defined]
    SwapGate._postconditions = lambda self, x1, x2, y1, y2: z3.And(  # type: ignore[attr-defined]
        x1 == y2, x2 == y1
    )
    CSwapGate._trivial_if = lambda self, x1, x2: x1 == x2  # type: ignore[attr-defined]
    CSwapGate._postconditions = lambda self, x1, x2, y1, y2: z3.And(  # type: ignore[attr-defined]
        x1 == y2, x2 == y1
    )
