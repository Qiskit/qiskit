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
Expectations (:mod:`qiskit.aqua.operators.expectations`)
====================================================================

.. currentmodule:: qiskit.aqua.operators.expectations

Expectations are converters which enable the computation of the expectation value of an
Observable with respect to some state function. They traverse an Operator tree, replacing
:class:`~qiskit.aqua.operators.state_fns.OperatorStateFn` measurements with equivalent
measurements which are more amenable to computation on quantum or classical hardware.
For example, if one would like to measure the
expectation value of an Operator ``o`` expressed as a sum of Paulis with respect to some state
function, but only has access to diagonal measurements on Quantum hardware, we can create a
measurement ~StateFn(o), use a :class:`PauliExpectation` to convert it to a diagonal measurement
and circuit pre-rotations to append to the state, and sample this circuit on Quantum hardware with
a :class:`~qiskit.aqua.operators.converters.CircuitSampler`. All in all, this would be:
``my_sampler.convert(my_expect.convert(~StateFn(o)) @ my_state).eval()``.

Expectation Base Class
======================
The ExpectationBase class gives an interface for algorithms to ask for Expectations as
execution settings. For example, if an algorithm contains an expectation value step within it,
such as :class:`~qiskit.aqua.algorithms.VQE`, the algorithm can give the opportunity for the user
to pass an ExpectationBase of their choice to be used in that expectation value step.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExpectationBase

Expectations
============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExpectationFactory
   AerPauliExpectation
   MatrixExpectation
   PauliExpectation
   CVaRExpectation

"""

from .expectation_base import ExpectationBase
from .expectation_factory import ExpectationFactory
from .pauli_expectation import PauliExpectation
from .aer_pauli_expectation import AerPauliExpectation
from .matrix_expectation import MatrixExpectation
from .cvar_expectation import CVaRExpectation

__all__ = ['ExpectationBase',
           'ExpectationFactory',
           'PauliExpectation',
           'AerPauliExpectation',
           'CVaRExpectation',
           'MatrixExpectation']
