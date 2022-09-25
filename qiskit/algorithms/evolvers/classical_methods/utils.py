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
"""Auxiliary functions for ScipyEvolvers"""
from typing import List, Tuple, Union

import scipy.sparse as sp
import numpy as np

from qiskit.quantum_info.states import Statevector
from qiskit import QuantumCircuit
from qiskit.opflow import StateFn

from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult

from ...list_or_dict import ListOrDict


def _create_observable_output(
    ops_ev_mean: np.ndarray,
    evolution_problem: EvolutionProblem,
) -> ListOrDict[Union[Tuple[np.ndarray, np.ndarray], Tuple[complex, complex], np.ndarray]]:
    """Creates the right output format for the evaluated auxiliary operators.
    Args:
        ops_ev_mean: Array containing the expectation value of each observable at each timestep.
        evolution_problem: Evolution Problem to create the output of.

    Returns:
        An output with the observables mean value at the appropiate times depending on whether
        the auxiliary operators in the evolution problem are a `list` or a `dict`.

    """

    aux_ops = evolution_problem.aux_operators

    time_array = np.linspace(0, evolution_problem.time, ops_ev_mean.shape[-1])
    zero_array = np.zeros(ops_ev_mean.shape[-1])  # std=0 since it is an exact method

    operator_number = 0 if aux_ops is None else len(aux_ops)

    observable_evolution = [(ops_ev_mean[i], zero_array) for i in range(operator_number)]

    if isinstance(aux_ops, dict):
        observable_evolution = dict(zip(aux_ops.keys(), observable_evolution))
        observable_evolution["time"] = time_array
    else:
        observable_evolution += [time_array]

    return observable_evolution


def _create_obs_final(
    ops_ev_mean: np.ndarray,
    evolution_problem: EvolutionProblem,
) -> ListOrDict[Tuple[complex, complex]]:
    """Creates the right output format for the final value of the auxiliary operators.

    Args:
        ops_ev_mean: Array containing the expectation value of each observable at the final timestep.
        evolution_problem: Evolution Problem to create the output of.

    Returns:
        An output with the observables mean value at the appropiate times depending on whether
        the auxiliary operators in the evolution problem are a `list` or a `dict`.

    """

    aux_ops = evolution_problem.aux_operators
    aux_ops_evaluated = [(op_ev, 0) for op_ev in ops_ev_mean]
    if isinstance(aux_ops, dict):
        aux_ops_evaluated = dict(zip(aux_ops.keys(), aux_ops_evaluated))
    return aux_ops_evaluated


def _evaluate_aux_ops(
    aux_ops: List[sp.csr_matrix],
    state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates the aux operators if they are provided and stores their value.

    Returns:
        Tuple of the mean and standard deviation of the aux operators for a given state.
    """
    op_mean = np.array([np.real(state.conjugate().dot(op.dot(state))) for op in aux_ops])
    return op_mean


def _sparsify(
    evolution_problem: EvolutionProblem, steps: int, real_time: bool
) -> Tuple[np.ndarray, List[sp.csr_matrix], sp.csr_matrix, float]:
    """Returns the matrices and parameters needed for time evolution in the appropiate format.

    Args:
        evolution_problem: The definition of the evolution problem.
        steps: number of timesteps to be performed.
        real_time: If `True` the operators returned will correspond to the ones for real time
            evolution, else they will correspond to imaginary time evolution.

    Returns:
        A tuple with the initial state, the list of operators to evaluate, the operator to be
        exponentiated to perfrom one timestep and its trace.
    """
    # Convert the initial state and Hamiltonian into sparse matrices.
    if isinstance(evolution_problem.initial_state, QuantumCircuit):
        state = Statevector(evolution_problem.initial_state).data.T
    else:
        state = evolution_problem.initial_state.to_matrix(massive=True).transpose()

    hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

    if isinstance(evolution_problem.aux_operators, list):
        aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
    elif isinstance(evolution_problem.aux_operators, dict):
        aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()]
    else:
        aux_ops = []
    timestep = evolution_problem.time / steps
    step_opeartor = -((1.0j) ** real_time) * timestep * hamiltonian
    step_operator_trace = step_opeartor.trace()
    return (state, aux_ops, step_opeartor, step_operator_trace)


def _evolve(evolution_problem: EvolutionProblem, steps: int, real_time: bool) -> EvolutionResult:
    r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

    Args:
        evolution_problem: The definition of the evolution problem.
        steps: number of timesteps to be performed.
        real_time: If `True` the operators returned will correspond to the ones for real time
            evolution, else they will correspond to imaginary time evolution.

    Returns:
        Evolution result which includes an evolved quantum state.

    Raises:
        ValueError: If the Hamiltonian is time dependent.

    """

    if evolution_problem.t_param is not None:
        raise ValueError("Time dependent Hamiltonians are not currently supported.")

    (state, aux_ops, step_opeartor, step_operator_trace) = _sparsify(
        evolution_problem=evolution_problem, steps=steps, real_time=real_time
    )

    # Create empty arrays to store the time evolution of the aux operators.
    number_operators = (
        0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
    )
    ops_ev_mean = np.empty(shape=(number_operators, steps + 1), dtype=complex)

    renormalize = (
        (lambda state: state) if real_time else (lambda state: state / np.linalg.norm(state))
    )

    # Perform the time evolution and stores the value of the operators at each timestep.
    for ts in range(steps):
        ops_ev_mean[:, ts] = _evaluate_aux_ops(aux_ops, state)
        state = sp.linalg.expm_multiply(A=step_opeartor, B=state, traceA=step_operator_trace)
        state = renormalize(state)

    ops_ev_mean[:, steps] = _evaluate_aux_ops(aux_ops, state)

    aux_ops_history = _create_observable_output(ops_ev_mean, evolution_problem)

    aux_ops = _create_obs_final(ops_ev_mean[:, -1], evolution_problem)

    return EvolutionResult(
        evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables=aux_ops_history
    )
