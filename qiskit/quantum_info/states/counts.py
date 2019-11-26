# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Routine for converting a state vector to counts.
"""
import numpy as np
from qiskit.exceptions import QiskitError


def state_to_counts(vec, eps=1e-15, density_matrix_diag=False):
    """Converts a statevector to counts
    of probabilities.

    This is useful, for example, when comparing
    output from the devices to the theoretically
    expected values using `plot_histogram` or
    the `hellinger_fidelity`.

    Parameters:
        vec (ndarray): Input statevector.
        eps (float): Optional tolerance.
        density_matrix_diag (bool): Input is a density matrix diagonal.

    Returns:
        dict: Counts of probabilities.

    Raises:
        QiskitError: Invalid input vector.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit, execute
           from qiskit.quantum_info.states import state_to_counts

           qc = QuantumCircuit(5)
           qc.h(2)
           qc.cx(2, 1)
           qc.cx(1, 0)
           qc.cx(2, 3)
           qc.cx(3, 4)

           sim = BasicAer.get_backend('statevector_simulator')
           res = execute(qc, sim).result()
           state_to_counts(res.get_statevector())
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise QiskitError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if not density_matrix_diag:
            val = val.real**2+val.imag**2
        if val > eps:
            counts[format(kk, str_format)] = val

    return counts
