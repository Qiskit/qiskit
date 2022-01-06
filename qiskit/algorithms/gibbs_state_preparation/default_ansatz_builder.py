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
"""Module for building a default ansatz for VarQITE-based Gibbs state preparation."""
import numpy as np
from numpy import ndarray

from qiskit.circuit.library import EfficientSU2


def build_ansatz(num_qubits: int, depth: int):
    """
    Builds a default ansatz for a VarQITE-based Gibbs state preparation which relies on EfficientSU2
    ansatz. Together with parameter values contructed by the method build_init_ansatz_params_vals,
    it results in num_qubits/2 maximally entangled states, as expected by the VarQITE-based Gibbs
    state preparation algorithm.
    Args:
        num_qubits: Number of qubits for an ansatz. Should be the same as the number of qubits of
                    a Hamiltonian that defines a desired Gibbs state.
        depth: Depth of an EfficientSU2 ansatz quantum circuit.
    Returns:
        A parametrized default ansatz for a VarQITE-based Gibbs state preparation algorithm.
    """
    entangler_map = [[i + 1, i] for i in range(num_qubits - 1)]
    ansatz = EfficientSU2(num_qubits, reps=depth, entanglement=entangler_map)
    qr = ansatz.qregs[0]
    non_aux_registers = int(len(qr) / 2)
    for i in range(non_aux_registers):
        ansatz.cx(qr[i], qr[i + non_aux_registers])
    return ansatz


def build_init_ansatz_params_vals(num_qubits: int, depth: int) -> ndarray:
    """
    Builds an array of default parameters for an ansatz constructed by the build_ansatz method.
    Together with that ansatz, it results in num_qubits/2 maximally entangled states, as expected
    by the VarQITE-based Gibbs state preparation algorithm.
    Args:
        num_qubits: Number of qubits for an ansatz. Should be the same as the number of qubits of
                    a Hamiltonian that defines a desired Gibbs state.
        depth: Depth of an EfficientSU2 ansatz quantum circuit.
    Returns:
        An array of parameters of length 2 * num_qubits * (depth + 1).
    """
    param_values_init = np.zeros(2 * num_qubits * (depth + 1))
    for j in range(2 * num_qubits * depth, int(len(param_values_init) - num_qubits - 2)):
        param_values_init[int(j)] = np.pi / 2.0
    return param_values_init
