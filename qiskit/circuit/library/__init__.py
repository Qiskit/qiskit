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

Generalized Gates
=================

.. autosummary::
   :toctree: ../stubs/

   Diagonal

Boolean Logic Circuits
======================

.. autosummary::
   :toctree: ../stubs/

   AND
   InnerProduct
   OR
   Permutation
   XOR

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
"""


from .boolean_logic import (
    Permutation,
    XOR,
    InnerProduct,
    OR,
    AND,
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
from .quantum_volume import QuantumVolume
from .diagonal import Diagonal
