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
State Functions (:mod:`qiskit.opflow.state_fns`)
================================================

.. deprecated:: 0.24.0

    The :mod:`qiskit.opflow` module is deprecated and will be removed no earlier
    than 3 months after the release date. For code migration guidelines,
    visit https://qisk.it/opflow_migration.

State functions are defined to be complex functions over a single binary
string (as compared to an operator, which is defined as a function over two binary strings,
or a function taking a binary function to another binary function). This function may be
called by the eval() method.

Measurements are defined to be functionals over StateFns, taking them to real values.
Generally, this real value is interpreted to represent the probability of some classical
state (binary string) being observed from a probabilistic or quantum system represented
by a StateFn. This leads to the equivalent definition, which is that a measurement m is
a function over binary strings producing StateFns, such that the probability of measuring
a given binary string b from a system with StateFn f is equal to the inner
product between f and m(b).

Note:
    All mathematical methods between StateFns are not in-place, meaning that they return a
    new object, but the underlying primitives are not copied.

Note:
    State functions here are not restricted to wave functions, as there is
    no requirement of normalization.

.. currentmodule:: qiskit.opflow.state_fns

State Functions
---------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   StateFn
   CircuitStateFn
   DictStateFn
   VectorStateFn
   SparseVectorStateFn
   OperatorStateFn
   CVaRMeasurement

"""

from .state_fn import StateFn
from .dict_state_fn import DictStateFn
from .operator_state_fn import OperatorStateFn
from .vector_state_fn import VectorStateFn
from .sparse_vector_state_fn import SparseVectorStateFn
from .circuit_state_fn import CircuitStateFn
from .cvar_measurement import CVaRMeasurement

__all__ = [
    "StateFn",
    "DictStateFn",
    "VectorStateFn",
    "CircuitStateFn",
    "OperatorStateFn",
    "CVaRMeasurement",
]
