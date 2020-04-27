# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Converters (:mod:`qiskit.aqua.operators.converters`)
====================================================
Converters are objects which manipulate Operators, usually traversing an Operator to
change certain sub-Operators into a desired representation. Often the converted Operator is
isomorphic or approximate to the original Operator in some way, but not always. For example,
a converter may accept ``CircuitOp`` and return a ``SummedOp`` of ``PauliOps`` representing the
circuit unitary. Converters may not have polynomial space or time scaling in their operations.
On the contrary, many converters, such as a ``MatrixExpectation`` or ``MatrixEvolution``,
which convert ``PauliOps`` to ``MatrixOps`` internally, will require time or space exponential
in the number of qubits unless a clever trick is known (such as the use of sparse matrices).

Note that not all converters are in this module, as Expectations and Evolutions are also
converters.

.. currentmodule:: qiskit.aqua.operators.converters

Converter Base Class
====================
The converter base class simply enforces the presence of a ``convert`` method.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ConverterBase

Converters
==========
In addition to the base class, directory holds a few miscellaneous converters which are used
frequently around the Operator flow.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CircuitSampler
   AbelianGrouper
   DictToCircuitSum
   PauliBasisChange

"""

from .converter_base import ConverterBase
from .circuit_sampler import CircuitSampler
from .pauli_basis_change import PauliBasisChange
from .dict_to_circuit_sum import DictToCircuitSum
from .abelian_grouper import AbelianGrouper

__all__ = ['ConverterBase',
           'CircuitSampler',
           'PauliBasisChange',
           'DictToCircuitSum',
           'AbelianGrouper']
