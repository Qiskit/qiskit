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
   TGate
   TdgGate
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

Particular Quantum Circuits
===========================

.. autosummary::
   :toctree: ../stubs/

   FourierChecking
   GraphState
   HiddenLinearFunction
   IQP
   QuantumVolume


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

Toffoli template circuits
=========================

.. autosummary::
   :toctree: ../stubs/

   template_circuits.toffoli.template_2a_1
   template_circuits.toffoli.template_2a_2
   template_circuits.toffoli.template_2a_3
   template_circuits.toffoli.template_4a_1
   template_circuits.toffoli.template_4a_2
   template_circuits.toffoli.template_4a_3
   template_circuits.toffoli.template_4a_4
   template_circuits.toffoli.template_4a_5
   template_circuits.toffoli.template_4b_1
   template_circuits.toffoli.template_4b_2
   template_circuits.toffoli.template_2a_2
   template_circuits.toffoli.template_5a_1
   template_circuits.toffoli.template_5a_2
   template_circuits.toffoli.template_5a_3
   template_circuits.toffoli.template_5a_4
   template_circuits.toffoli.template_6a_1
   template_circuits.toffoli.template_6a_2
   template_circuits.toffoli.template_6a_3
   template_circuits.toffoli.template_6a_4
   template_circuits.toffoli.template_6b_1
   template_circuits.toffoli.template_6b_2
   template_circuits.toffoli.template_6c_1
   template_circuits.toffoli.template_7a_1
   template_circuits.toffoli.template_7b_1
   template_circuits.toffoli.template_7c_1
   template_circuits.toffoli.template_7d_1
   template_circuits.toffoli.template_7e_1
   template_circuits.toffoli.template_2a_1
   template_circuits.toffoli.template_9a_1
   template_circuits.toffoli.template_9c_1
   template_circuits.toffoli.template_9c_2
   template_circuits.toffoli.template_9c_3
   template_circuits.toffoli.template_9c_4
   template_circuits.toffoli.template_9c_5
   template_circuits.toffoli.template_9c_6
   template_circuits.toffoli.template_9c_7
   template_circuits.toffoli.template_9c_8
   template_circuits.toffoli.template_9c_9
   template_circuits.toffoli.template_9c_10
   template_circuits.toffoli.template_9c_11
   template_circuits.toffoli.template_9c_12
   template_circuits.toffoli.template_9d_1
   template_circuits.toffoli.template_9d_2
   template_circuits.toffoli.template_9d_3
   template_circuits.toffoli.template_9d_4
   template_circuits.toffoli.template_9d_5
   template_circuits.toffoli.template_9d_6
   template_circuits.toffoli.template_9d_7
   template_circuits.toffoli.template_9d_8
   template_circuits.toffoli.template_9d_9
   template_circuits.toffoli.template_9d_10

"""

from .standard_gates import *
from .template_circuits import *
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
