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
from typing import Union, Dict, Iterable, Optional

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import CircuitSampler, StateFn, OperatorBase
from qiskit.utils import QuantumInstance
from qiskit.providers import BaseBackend


# TODO change name, refactor expected val calculation, will be used for error bounds
def _calculate_distance_energy(
    state,
    h_matrix,
    param_dict: Dict[Parameter, Union[float, complex]],
    state_circ_sampler: Optional[CircuitSampler] = None,
) -> float:
    """
    Evaluate the fidelity to the target state, the energy w.r.t. the target state and
    the energy w.r.t. the trained state for a given time and the current parameter set
    Args:
        state: Trained state.
        h_matrix: Hamiltonian matrix.
        param_dict: Dictionary which matches the operator parameters to the current
        values of parameters for the given time.
        state_circ_sampler: Optional sampler for the state.
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

    # Trained Energy
    trained_en = _inner_prod(trained_state, np.dot(h_matrix, trained_state))

    return np.real(trained_en)


def _inner_prod(x: Iterable, y: Iterable) -> Union[np.ndarray, np.complex, np.float]:
    """
    Compute the inner product of two vectors
    Args:
        x: vector
        y: vector
    Returns: Inner product of x,y
    """
    return np.matmul(np.conj(np.transpose(x)), y)


# TODO not used anywhere
def calc_energy(
    hamiltonian: OperatorBase,
    ansatz: QuantumCircuit,
    param_dict: Dict,
    backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
) -> float:
    """
    Compute energy for a given Hamiltonian, ansatz and parameter dictionary.
    Args:
        hamiltonian: System hamiltonian.
        ansatz: Parameterized ansatz.
        param_dict: Parameter dictionary.
        backend: Backend used for energy calculation.

    Returns: Energy

    """
    energy = ~StateFn(hamiltonian) @ StateFn(ansatz)
    if backend is not None:
        energy = CircuitSampler(backend).convert(energy)
    energy_val = energy.assign_parameters(param_dict).eval()
    return energy_val
