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

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn
from ..list_or_dict import ListOrDict
from ...quantum_info import SparsePauliOp


class EvolutionProblem:
    """Evolution problem class.

    This class is the input to time evolution algorithms and must contain information on the total
    evolution time, a quantum state to be evolved and under which Hamiltonian the state is evolved.
    """

    def __init__(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Union[StateFn, QuantumCircuit],
        aux_operators: Optional[ListOrDict[OperatorBase]] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[complex]]] = None,
    ):
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                evolved ``initial_state`` and their expectation values returned.
            t_param: Time parameter in case of a time-dependent Hamiltonian. This
                free parameter must be within the ``hamiltonian``.
            hamiltonian_value_dict: If the Hamiltonian contains free parameters, this
                dictionary maps all these parameters to values.

        Raises:
            ValueError: If non-positive time of evolution is provided.
            ValueError: If no ``initial_state`` is provided.
            ValueError: If not all parameter values are provided.
        """
        if time <= 0:
            raise ValueError(f"Time of evolution provided is not positive, detected time={time}.")
        if initial_state is None:
            raise ValueError("No initial_state provided for the EvolutionProblem. It is required.")
        # TODO SparsePauliOp does not have .parameters because it is not allowed to be parametrized.
        #  Can we handle this better than with an if?
        if not isinstance(hamiltonian, SparsePauliOp):
            self._check_parameters(hamiltonian, hamiltonian_value_dict, t_param)
        self.hamiltonian = hamiltonian
        self.time = time
        self.initial_state = initial_state
        self.aux_operators = aux_operators
        self.t_param = t_param
        self.hamiltonian_value_dict = hamiltonian_value_dict

    def _check_parameters(
        self,
        hamiltonian: OperatorBase,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[complex]]] = None,
        t_param: Optional[Parameter] = None,
    ) -> None:
        """
        Checks if all parameters present in the Hamiltonian are also present in the dictionary
        that maps them to values.
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system.
            hamiltonian_value_dict: If the Hamiltonian contains free parameters, this
                dictionary maps all these parameters to values.
            t_param: Time parameter in case of a time-dependent Hamiltonian. This
            free parameter must be within the ``hamiltonian``.

        Raises:
            ValueError: If there are unbound parameters in the Hamiltonian.
        """
        t_param_set = set()
        if t_param is not None:
            t_param_set.add(t_param)
        hamiltonian_dict_param_set = set()
        if hamiltonian_value_dict is not None:
            hamiltonian_dict_param_set.union(set(hamiltonian_value_dict.keys()))
        params_set = t_param_set.union(hamiltonian_dict_param_set)
        hamiltonian_param_set = set(hamiltonian.parameters)
        if hamiltonian_param_set != params_set:
            raise ValueError(
                f"Provided parameters {params_set} do not match Hamiltonian parameters "
                f"{hamiltonian_param_set}."
            )
