# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" sub system """

from collections import defaultdict
import numpy as np
from scipy.linalg import sqrtm

from qiskit.tools.qi.qi import partial_trace


def get_subsystem_density_matrix(statevector, trace_systems):
    """
    Compute the reduced density matrix of a quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced out.

    Returns:
        numpy.ndarray: The reduced density matrix for the desired subsystem
    """
    rho = np.outer(statevector, np.conj(statevector))
    rho_sub = partial_trace(rho, trace_systems)
    return rho_sub


def get_subsystem_fidelity(statevector, trace_systems, subsystem_state):
    """
    Compute the fidelity of the quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced.
            to trace qubits 0 and 4 trace_systems = [0,4]
        subsystem_state (list|array): The ground-truth state vector of the subsystem

    Returns:
        numpy.ndarray: The subsystem fidelity
    """
    rho = np.outer(np.conj(statevector), statevector)
    rho_sub = partial_trace(rho, trace_systems)
    rho_sub_in = np.outer(np.conj(subsystem_state), subsystem_state)
    fidelity = np.trace(
        sqrtm(
            np.dot(
                np.dot(sqrtm(rho_sub), rho_sub_in),
                sqrtm(rho_sub)
            )
        )
    ) ** 2
    return fidelity


def get_subsystems_counts(complete_system_counts):
    """
    Extract all subsystems' counts from the single complete system count dictionary.

    If multiple classical registers are used to measure various parts of a quantum system,
    Each of the measurement dictionary's keys would contain spaces as delimiters to separate
    the various parts being measured. For example, you might have three keys
    '11 010', '01 011' and '11 011', among many other, in the count dictionary of the
    5-qubit complete system, and would like to get the two subsystems' counts
    (one 2-qubit, and the other 3-qubit) in order to get the counts for the 2-qubit
    partial measurement '11' or the 3-qubit partial measurement '011'.


    Args:
        complete_system_counts (dict): The measurement count dictionary of a complete system
            that contains multiple classical registers for measurements s.t. the dictionary's
            keys have space delimiters.

    Returns:
        list: A list of measurement count dictionaries corresponding to
                each of the subsystems measured.
    """
    mixed_measurements = list(complete_system_counts)
    subsystems_counts = [defaultdict(int) for _ in mixed_measurements[0].split()]
    for mixed_measurement in mixed_measurements:
        count = complete_system_counts[mixed_measurement]
        for k, d_l in zip(mixed_measurement.split(), subsystems_counts):
            d_l[k] += count
    return [dict(d) for d in subsystems_counts]
