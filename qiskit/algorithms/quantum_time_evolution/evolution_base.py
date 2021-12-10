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

"""Base class for quantum time evolution."""

from abc import ABC, abstractmethod
from typing import Union, List

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn, Gradient


class EvolutionBase(ABC):
    """Base class for quantum time evolution."""

    @abstractmethod
    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn = None,
        observable: OperatorBase = None,
        t_param: Parameter = None,
        hamiltonian_value_dict: [Parameter, Union[float, complex]] = None,
    ):
        """
        Evolves an initial state according to a Hamiltonian provided.
        Args:
            hamiltonian:
                ⟨ψ(ω)|H|ψ(ω)〉
                Operator used vor time evolution.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                                    certain values, including the t_param.
                                    # TODO do we allow binding t_param here?
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
        t_param: Parameter = None,
        hamiltonian_value_dict: [Parameter, Union[float, complex]] = None,
        gradient_params: List[Parameter] = None,
    ):
        """Performs Quantum Time Evolution of gradient expressions."""
        raise NotImplementedError()
