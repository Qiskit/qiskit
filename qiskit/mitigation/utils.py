# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout mitigation data handling utils
"""

from typing import Optional, List, Dict, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, marginal_counts

def counts_probability_vector(
        counts: Counts,
        qubits: Optional[List[int]] = None,
        clbits: Optional[List[int]] = None,
        num_qubits: Optional[int] = None,
        return_shots: Optional[bool] = False) -> np.ndarray:
    """Compute mitigated expectation value.

    Args:
        counts: counts object
        qubits: qubits the count bitstrings correspond to.
        clbits: Optional, marginalize counts to just these bits.
        num_qubits: the total number of qubits.
        return_shots: return the number of shots.

    Raises:
        QiskitError: if qubit and clbit kwargs are not valid.

    Returns:
        np.ndarray: a probability vector for all count outcomes.
    """
    qubits_len = len(qubits) if not qubits is None else 0
    clbits_len = len(clbits) if not clbits is None else 0
    if clbits_len != 0 and qubits_len != clbits_len:
        raise QiskitError("Num qubits ({}) does not match number of clbits ({}).".format(qubits_len, clbits_len))

    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, clbits)

    # Get total number of qubits
    if num_qubits is None:
        num_qubits = len(next(iter(counts)))

    # Get vector
    vec = np.zeros(2**num_qubits, dtype=float)
    shots = 0
    for key, val in counts.items():
        shots += val
        vec[int(key, 2)] = val
    vec /= shots

    # Remap qubits
    if qubits is not None:
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match vector length.")
        axes = [num_qubits - 1 - i for i in reversed(np.argsort(qubits))]
        vec = np.reshape(vec,
                         num_qubits * [2]).transpose(axes).reshape(vec.shape)
    if return_shots:
        return vec, shots
    return vec