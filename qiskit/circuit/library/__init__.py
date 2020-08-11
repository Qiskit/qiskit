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
   DCXGate
   CHGate
   CPhaseGate
   CRXGate
   CRYGate
   CRZGate
   CSwapGate
   CSXGate
   CUGate
   CU1Gate
   CU3Gate
   CXGate
   CYGate
   CZGate
   HGate
   IGate
   MCPhaseGate
   MCXGate
   MCXGrayCode
   MCXRecursive
   MCXVChain
   Measure
   MSGate
   PhaseGate
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
   SXGate
   SXdgGate
   TGate
   TdgGate
   UGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate

Generalized Gates
=================

.. autosummary::
   :toctree: ../stubs/

   Diagonal
   MCMT
   MCMTVChain
   Permutation
   GMS

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

Functions on binary variables
+++++++++++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/

   QuadraticForm

Particular Quantum Circuits
===========================

.. autosummary::
   :toctree: ../stubs/

   FourierChecking
   GraphState
   HiddenLinearFunction
   IQP
   QuantumVolume
   PhaseEstimation


N-local circuits
================

.. autosummary::
   :toctree: ../stubs/

   NLocal
   TwoLocal
   RealAmplitudes
   EfficientSU2
   ExcitationPreserving


Data encoding circuits
======================

.. autosummary::
   :toctree: ../stubs/

   PauliFeatureMap
   ZFeatureMap
   ZZFeatureMap

"""

from .standard_gates import *
from ..barrier import Barrier
from ..measure import Measure
from ..reset import Reset

from .blueprintcircuit import BlueprintCircuit
from .generalized_gates import (
    Diagonal,
    MCMT,
    MCMTVChain,
    Permutation,
    GMS
)
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
    QuadraticForm,
)
from .n_local import (
    NLocal,
    TwoLocal,
    RealAmplitudes,
    EfficientSU2,
    ExcitationPreserving,
)
from .data_preparation import (
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap
)
from .quantum_volume import QuantumVolume
from .fourier_checking import FourierChecking
from .graph_state import GraphState
from .hidden_linear_function import HiddenLinearFunction
from .iqp import IQP
from .phase_estimation import PhaseEstimation
