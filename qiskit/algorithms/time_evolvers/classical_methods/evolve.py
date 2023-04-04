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
"""Auxiliary functions for SciPy Time Evolvers"""
from __future__ import annotations
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import numpy as np

from qiskit.quantum_info.states import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from ..time_evolution_problem import TimeEvolutionProblem
from ..time_evolution_result import TimeEvolutionResult
from ...exceptions import AlgorithmError

from ...list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


def _create_observable_output(
    ops_ev_mean: np.ndarray,
    evolution_problem: TimeEvolutionProblem,
) -> tuple[ListOrDict[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Creates the right output format for the evaluated auxiliary operators.
    Args:
        ops_ev_mean: Array containing the expectation value of each observable at each timestep.
        evolution_problem: Time Evolution Problem to create the output of.

    Returns:
        An output with the observables mean value at the appropriate times depending on whether
        the auxiliary operators in the time evolution problem are a `list` or a `dict`.

    """

    aux_ops = evolution_problem.aux_operators

    time_array = np.linspace(0, evolution_problem.time, ops_ev_mean.shape[-1])
    zero_array = np.zeros(ops_ev_mean.shape[-1])  # std=0 since it is an exact method

    operators_number = 0 if aux_ops is None else len(aux_ops)

    observable_evolution = [(ops_ev_mean[i], zero_array) for i in range(operators_number)]

    if isinstance(aux_ops, dict):
        observable_evolution = dict(zip(aux_ops.keys(), observable_evolution))

    return observable_evolution, time_array


def _create_obs_final(
    ops_ev_mean: np.ndarray,
    evolution_problem: TimeEvolutionProblem,
) -> ListOrDict[tuple[complex, complex]]:
    """Creates the right output format for the final value of the auxiliary operators.

    Args:
        ops_ev_mean: Array containing the expectation value of each observable at the final timestep.
        evolution_problem: Evolution problem to create the output of.

    Returns:
        An output with the observables mean value at the appropriate times depending on whether
        the auxiliary operators in the evolution problem are a `list` or a `dict`.

    """

    aux_ops = evolution_problem.aux_operators
    aux_ops_evaluated: ListOrDict[tuple[complex, complex]] = [(op_ev, 0) for op_ev in ops_ev_mean]
    if isinstance(aux_ops, dict):
        aux_ops_evaluated = dict(zip(aux_ops.keys(), aux_ops_evaluated))
    return aux_ops_evaluated


def _evaluate_aux_ops(
    aux_ops: list[csr_matrix],
    state: np.ndarray,
) -> np.ndarray:
    """Evaluates the aux operators if they are provided and stores their value.

    Returns:
        Mean of the aux operators for a given state.
    """
    op_means = np.array([np.real(state.conjugate().dot(op.dot(state))) for op in aux_ops])
    return op_means


def _operator_to_matrix(operator: BaseOperator | PauliSumOp):

    if isinstance(operator, PauliSumOp):
        op_matrix = operator.to_spmatrix()
    else:
        try:
            op_matrix = operator.to_matrix(sparse=True)
        except TypeError:
            logger.debug(
                "WARNING: operator of type `%s` does not support sparse matrices. "
                "Trying dense computation",
                type(operator),
            )
            try:
                op_matrix = operator.to_matrix()
            except AttributeError as ex:
                raise AlgorithmError(f"Unsupported operator type `{type(operator)}`.") from ex
    return op_matrix


def _build_scipy_operators(
    evolution_problem: TimeEvolutionProblem, num_timesteps: int, real_time: bool
) -> tuple[np.ndarray, list[csr_matrix], csr_matrix]:
    """Returns the matrices and parameters needed for time evolution in the appropriate format.

    Args:
        evolution_problem: The definition of the evolution problem.
        num_timesteps: Number of timesteps to be performed.
        real_time: If `True`, returned operators will correspond to real time evolution,
            Else, they will correspond to imaginary time evolution.

    Returns:
        A tuple with the initial state, the list of operators to evaluate and the operator to be
        exponentiated to perform one timestep.

    Raises:
        ValueError: If the Hamiltonian can not be converted into a sparse matrix or dense matrix.
    """
    # Convert the initial state and Hamiltonian into sparse matrices.
    if isinstance(evolution_problem.initial_state, QuantumCircuit):
        state = Statevector(evolution_problem.initial_state).data
    else:
        state = evolution_problem.initial_state.data

    hamiltonian = _operator_to_matrix(operator=evolution_problem.hamiltonian)

    if isinstance(evolution_problem.aux_operators, list):
        aux_ops = [
            _operator_to_matrix(operator=aux_op) for aux_op in evolution_problem.aux_operators
        ]
    elif isinstance(evolution_problem.aux_operators, dict):
        aux_ops = [
            _operator_to_matrix(operator=aux_op)
            for aux_op in evolution_problem.aux_operators.values()
        ]
    else:
        aux_ops = []
    timestep = evolution_problem.time / num_timesteps
    step_operator = -((1.0j) ** real_time) * timestep * hamiltonian
    return state, aux_ops, step_operator


def _evolve(
    evolution_problem: TimeEvolutionProblem, num_timesteps: int, real_time: bool
) -> TimeEvolutionResult:
    r"""Performs either real  or imaginary time evolution :math:`\exp(-i t H)|\Psi\rangle`.

    Args:
        evolution_problem: The definition of the evolution problem.
        num_timesteps: Number of timesteps to be performed.
        real_time: If `True`, returned operators will correspond to real time evolution,
            Else, they will correspond to imaginary time evolution.

    Returns:
        Evolution result which includes an evolved quantum state.

    Raises:
        ValueError: If the Hamiltonian is time dependent.
        ValueError: If the initial state is `None`.

    """
    if num_timesteps <= 0:
        raise ValueError("Variable `num_timesteps` needs to be a positive integer.")

    if evolution_problem.t_param is not None:
        raise ValueError("Time dependent Hamiltonians are not supported.")

    if evolution_problem.initial_state is None:
        raise ValueError("Initial state is `None`")

    state, aux_ops, step_operator = _build_scipy_operators(
        evolution_problem=evolution_problem, num_timesteps=num_timesteps, real_time=real_time
    )

    # Create empty arrays to store the time evolution of the aux operators.
    number_operators = (
        0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
    )
    ops_ev_mean = np.empty(shape=(number_operators, num_timesteps + 1), dtype=complex)

    renormalize = (
        (lambda state: state) if real_time else (lambda state: state / np.linalg.norm(state))
    )

    # Perform the time evolution and stores the value of the operators at each timestep.
    for ts in range(num_timesteps):
        ops_ev_mean[:, ts] = _evaluate_aux_ops(aux_ops, state)
        state = expm_multiply(A=step_operator, B=state)
        state = renormalize(state)

    ops_ev_mean[:, num_timesteps] = _evaluate_aux_ops(aux_ops, state)

    observable_history, times = _create_observable_output(ops_ev_mean, evolution_problem)
    aux_ops_evaluated = _create_obs_final(ops_ev_mean[:, -1], evolution_problem)

    return TimeEvolutionResult(
        evolved_state=Statevector(state),
        aux_ops_evaluated=aux_ops_evaluated,
        observables=observable_history,
        times=times,
    )
