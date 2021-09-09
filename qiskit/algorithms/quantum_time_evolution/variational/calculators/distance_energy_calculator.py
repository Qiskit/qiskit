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
from typing import Union, Dict, Iterable

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.calculators.bures_distance_calculator import (
    _calculate_bures_distance,
)
from qiskit.quantum_info import state_fidelity


def _calculate_distance_energy(
    state,
    exact_state,
    h_matrix,
    param_dict: Dict,
    state_circ_sampler=None,
) -> (float, float, float):
    """
    Evaluate the fidelity to the target state, the energy w.r.t. the target state and
    the energy w.r.t. the trained state for a given time and the current parameter set
    Args:
        time: current evolution time
        param_dict: dictionary which matches the operator parameters to the current
        values of parameters for the given time
    Returns: fidelity to the target state, the energy w.r.t. the target state and
    the energy w.r.t. the trained state
    """

    # |state_t>
    if state_circ_sampler is not None:
        trained_state = state_circ_sampler.convert(state, params=param_dict)
    else:
        # TODO state was already bound earlier, error
        trained_state = state.assign_parameters(param_dict)
    trained_state = trained_state.eval().primitive.data
    target_state = exact_state

    # Fidelity
    f = state_fidelity(target_state, trained_state)
    # Actual error
    act_err = np.linalg.norm(np.subtract(target_state, trained_state), ord=2)
    phase_agnostic_act_err = _calculate_bures_distance(target_state, trained_state)
    # Target Energy
    act_en = _inner_prod(target_state, np.dot(h_matrix, target_state))
    # Trained Energy
    trained_en = _inner_prod(trained_state, np.dot(h_matrix, trained_state))

    return f, act_err, phase_agnostic_act_err, np.real(act_en), np.real(trained_en)


def _inner_prod(x: Iterable, y: Iterable) -> Union[np.ndarray, np.complex, np.float]:
    """
    Compute the inner product of two vectors
    Args:
        x: vector
        y: vector
    Returns: Inner product of x,y
    """
    return np.matmul(np.conj(np.transpose(x)), y)
