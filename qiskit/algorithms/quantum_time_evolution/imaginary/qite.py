# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Interface for Quantum Imaginary Time Evolution."""

from abc import abstractmethod

from qiskit.algorithms.quantum_time_evolution.evolution_base import EvolutionBase
from qiskit.opflow import OperatorBase, StateFn, Gradient


class Qite(EvolutionBase):
    """Interface for Quantum Imaginary Time Evolution."""

    @abstractmethod
    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn = None,
        observable: OperatorBase = None,
        t_param=None,
        hamiltonian_value_dict=None,
    ):
        """
        Performs Quantum Imaginary Time Evolution on an initial state according to a Hamiltonian
        provided.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: OperatorBase = None,
        t_param=None,
        hamiltonian_value_dict=None,
        gradient_params=None,
    ):
        """Performs Quantum Imaginary Time Evolution of gradient expressions."""
        raise NotImplementedError()
