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
from abc import ABC
from typing import List, Tuple, Union

import scipy.sparse as sp
import numpy as np

from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem

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
