# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Converters (:mod:`qiskit.opflow.converters`)
============================================

.. currentmodule:: qiskit.opflow.converters

.. deprecated:: 0.24.0

    The :mod:`qiskit.opflow` module is deprecated and will be removed no earlier
    than 3 months after the release date. For code migration guidelines,
    visit https://qisk.it/opflow_migration.

Converters are objects which manipulate Operators, usually traversing an Operator to
change certain sub-Operators into a desired representation. Often the converted Operator is
isomorphic or approximate to the original Operator in some way, but not always. For example,
a converter may accept :class:`~qiskit.opflow.primitive_ops.CircuitOp` and return a
:class:`~qiskit.opflow.list_ops.SummedOp` of
:class:`~qiskit.opflow.primitive_ops.PauliOp`'s representing the
circuit unitary. Converters may not have polynomial space or time scaling in their operations.
On the contrary, many converters, such as a
:class:`~qiskit.opflow.expectations.MatrixExpectation` or
:class:`~qiskit.opflow.evolutions.MatrixEvolution`,
which convert :class:`~qiskit.opflow.primitive_ops.PauliOp`'s to
:class:`~qiskit.opflow.primitive_ops.MatrixOp`'s internally, will require time or space
exponential in the number of qubits unless a clever trick is known
(such as the use of sparse matrices).


Note:
     Not all converters are in this module, as :mod:`~qiskit.opflow.expectations`
     and :mod:`~qiskit.opflow.evolutions` are also converters.

Converter Base Class
--------------------
The converter base class simply enforces the presence of a :meth:`~ConverterBase.convert` method.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   ConverterBase

Converters
----------
In addition to the base class, directory holds a few miscellaneous converters which are used
frequently around the Operator flow.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   CircuitSampler
   AbelianGrouper
   DictToCircuitSum
   PauliBasisChange
   TwoQubitReduction
"""

from .converter_base import ConverterBase
from .circuit_sampler import CircuitSampler
from .pauli_basis_change import PauliBasisChange
from .dict_to_circuit_sum import DictToCircuitSum
from .abelian_grouper import AbelianGrouper
from .two_qubit_reduction import TwoQubitReduction

__all__ = [
    "ConverterBase",
    "CircuitSampler",
    "PauliBasisChange",
    "DictToCircuitSum",
    "AbelianGrouper",
    "TwoQubitReduction",
]
