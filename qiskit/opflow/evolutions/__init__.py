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
Operator Evolutions (:mod:`qiskit.opflow.evolutions`)
=====================================================

.. currentmodule:: qiskit.opflow.evolutions

.. deprecated:: 0.24.0

    The :mod:`qiskit.opflow` module is deprecated and will be removed no earlier
    than 3 months after the release date. For code migration guidelines,
    visit https://qisk.it/opflow_migration.

Evolutions are converters which traverse an Operator tree, replacing
any :class:`EvolvedOp` `e` with a Schrodinger equation-style evolution
:class:`~qiskit.opflow.primitive_ops.CircuitOp`
equalling or approximating the matrix exponential of -i * the Operator contained inside
(`e.primitive`). The Evolutions are essentially implementations of Hamiltonian Simulation
algorithms, including various methods for Trotterization.

The :class:`EvolvedOp` is simply a placeholder signifying that the Operator inside it should be
converted to its exponential by the Evolution converter. All Operators
(not :mod:`~qiskit.opflow.state_fns`) have
``.exp_i()`` methods which either return the exponential of the Operator directly,
or an :class:`EvolvedOp` containing the Operator.


Note:
    Evolutions work with parameterized Operator coefficients, so
    ``my_expectation.convert((t * H).exp_i())``, where t is a scalar or Terra Parameter and H
    is an Operator, will produce a :class:`~qiskit.opflow.primitive_ops.CircuitOp`
    equivalent to e^iHt.

Evolution Base Class
--------------------

The EvolutionBase class gives an interface for algorithms to ask for Evolutions as
execution settings. For example, if an algorithm contains an Operator evolution step within it,
such as :class:`~qiskit.algorithms.QAOA`, the algorithm can give the opportunity for the user
to pass an EvolutionBase of their choice to be used in that evolution step.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   EvolutionBase

Evolutions
----------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   EvolutionFactory
   EvolvedOp
   MatrixEvolution
   PauliTrotterEvolution

Trotterizations
---------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   TrotterizationBase
   TrotterizationFactory
   Trotter
   Suzuki
   QDrift
"""

from .evolution_base import EvolutionBase
from .evolution_factory import EvolutionFactory
from .evolved_op import EvolvedOp
from .pauli_trotter_evolution import PauliTrotterEvolution
from .matrix_evolution import MatrixEvolution
from .trotterizations import TrotterizationBase, TrotterizationFactory, Trotter, Suzuki, QDrift

# TODO co-diagonalization of Abelian groups in PauliTrotterEvolution
# TODO quantum signal processing/qubitization
# TODO evolve by density matrix (need to add iexp to operator_state_fn)
# TODO linear combination evolution

__all__ = [
    "EvolutionBase",
    "EvolutionFactory",
    "EvolvedOp",
    "PauliTrotterEvolution",
    "MatrixEvolution",
    "TrotterizationBase",
    "TrotterizationFactory",
    "Trotter",
    "Suzuki",
    "QDrift",
]
