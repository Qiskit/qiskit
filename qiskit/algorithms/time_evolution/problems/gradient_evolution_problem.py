# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gradient evolution problem class."""

from typing import Union, List, Optional, Dict

from qiskit.algorithms.time_evolution.problems.evolution_problem import EvolutionProblem
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn, Gradient


class GradientEvolutionProblem(EvolutionProblem):
    """Gradient evolution problem class."""

    def __init__(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
        gradient_params: Optional[List[Parameter]] = None,
    ):
        """
        Args:
            hamiltonian: Operator used for variational time evolution.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            gradient_object: Gradient object which defines a method for computing desired
                gradients.
            observable: Observable to be evolved.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                certain values, including the t_param.
            gradient_params: List of parameters that indicates with respect to which parameters
                gradients shall be computed.
        """
        super().__init__(
            hamiltonian, time, initial_state, observable, t_param, hamiltonian_value_dict
        )

        self.gradient_object = gradient_object
        self.gradient_params = gradient_params
