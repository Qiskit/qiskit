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

"""Evolution problem class."""

from typing import Union, Optional, Dict

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn


class EvolutionProblem:
    """Evolution problem class."""

    def __init__(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Optional[StateFn] = None,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
    ):
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved; mutually exclusive with observable.
            observable: Observable to be evolved; mutually exclusive with initial_state.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to certain
                values, including the t_param.

        Raises:
            ValueError: If both or none initial_state and observable are provided.
        """

        if initial_state is not None and observable is not None:
            raise ValueError("initial_state and observable are mutually exclusive; both provided.")

        if initial_state is None and observable is None:
            raise ValueError("One of initial_state or observable must be provided; none provided.")

        self.hamiltonian = hamiltonian
        self.time = time
        self.initial_state = initial_state
        self.observable = observable
        self.t_param = t_param
        self.hamiltonian_value_dict = hamiltonian_value_dict
