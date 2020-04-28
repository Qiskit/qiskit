# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===============================================
Circuit Library (:mod:`qiskit.circuit.library`)
===============================================

.. currentmodule:: qiskit.circuit.library

Standard Gates
==============

.. autosummary::
   :toctree: ../stubs/

   Barrier
   C3XGate
   C4XGate
   CCXGate
   CHGate
   CRXGate
   CRYGate
   CRZGate
   CSwapGate
   CU1Gate
   CU3Gate
   CXGate
   CYGate
   CZGate
   HGate
   IGate
   Measure
   MSGate
   RCCXGate
   RC3XGate
   Reset
   RXGate
   RXXGate
   RYGate
   RYYGate
   RZGate
   RZZGate
   RZXGate
   SGate
   SdgGate
   SwapGate
   iSwapGate
   DCXGate
   TdgGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate


Blueprint Circuit
=================

.. autosummary::
   :toctree: ../stubs/

   BlueprintCircuit

Generalized Gates
=================

.. autosummary::
   :toctree: ../stubs/

   Diagonal
   Permutation

Boolean Logic Circuits
======================

.. autosummary::
   :toctree: ../stubs/

   AND
   OR
   XOR
   InnerProduct

Basis Change Circuits
=====================

.. autosummary::
   :toctree: ../stubs/

   QFT

Arithmetic Circuits
===================

Functional Pauli Rotations
++++++++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/

   FunctionalPauliRotations
   LinearPauliRotations
   PolynomialPauliRotations
   PiecewiseLinearPauliRotations

Adders
++++++

.. autosummary::
   :toctree: ../stubs/

   WeightedAdder

Comparators
+++++++++++

.. autosummary::
   :toctree: ../stubs/

   IntegerComparator

Characterization and Validation Circuits
========================================

.. autosummary::
   :toctree: ../stubs/

   QuantumVolume

Quantum Complexity Advantage Circuits
=====================================

.. autosummary::
   :toctree: ../stubs/

   FourierChecking


N-local circuits
================

.. autosummary::
   :toctree: ../stubs/

   NLocal
   TwoLocal
   RY
   RYRZ
   SwapRZ


Data encoding circuits
======================

.. autosummary::
   :toctree: ../stubs/

   PauliExpansion
   FirstOrderExpansion
   SecondOrderExpansion

"""

from .standard_gates import *
from ..barrier import Barrier
from ..measure import Measure
from ..reset import Reset

from .blueprintcircuit import BlueprintCircuit
from .generalized_gates import Diagonal
from .generalized_gates import Permutation
from .boolean_logic import (
    AND,
    OR,
    XOR,
    InnerProduct,
)
from .basis_change import QFT
from .arithmetic import (
    FunctionalPauliRotations,
    LinearPauliRotations,
    PiecewiseLinearPauliRotations,
    PolynomialPauliRotations,
    IntegerComparator,
    WeightedAdder,
)
from .fourier_checking import FourierChecking
from .n_local import (
    NLocal,
    TwoLocal,
    RY,
    RYRZ,
    SwapRZ,
)
from .data_encoding import (
    PauliExpansion,
    FirstOrderExpansion,
    SecondOrderExpansion
)
from .quantum_volume import QuantumVolume
