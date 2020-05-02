# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Legacy Operators (:mod:`qiskit.aqua.operators.legacy`)
======================================================

.. currentmodule:: qiskit.aqua.operators.legacy

These are the Operators provided by Aqua up until the 0.6 release. These are being replaced
by the operator flow function and we encourage you to use this.

Note:
    At some future time this legacy operator logic will be deprecated and removed.

Legacy Operators
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LegacyBaseOperator
   WeightedPauliOperator
   TPBGroupedWeightedPauliOperator
   MatrixOperator

Legacy Operator support
=======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    evolution_instruction
    suzuki_expansion_slice_pauli_list
    pauli_measurement
    measure_pauli_z
    covariance
    row_echelon_F2
    kernel_F2
    commutator
    check_commutativity
    PauliGraph
    Z2Symmetries
"""
from .common import (evolution_instruction, suzuki_expansion_slice_pauli_list, pauli_measurement,
                     measure_pauli_z, covariance, row_echelon_F2,
                     kernel_F2, commutator, check_commutativity)

from .base_operator import LegacyBaseOperator
from .weighted_pauli_operator import WeightedPauliOperator, Z2Symmetries
from .matrix_operator import MatrixOperator
from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
from .pauli_graph import PauliGraph

__all__ = [
    'evolution_instruction',
    'suzuki_expansion_slice_pauli_list',
    'pauli_measurement',
    'measure_pauli_z',
    'covariance',
    'row_echelon_F2',
    'kernel_F2',
    'commutator',
    'check_commutativity',
    'PauliGraph',
    'LegacyBaseOperator',
    'WeightedPauliOperator',
    'Z2Symmetries',
    'TPBGroupedWeightedPauliOperator',
    'MatrixOperator'
]
