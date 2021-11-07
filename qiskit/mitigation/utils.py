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

import logging
from typing import Optional, List, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, marginal_counts

logger = logging.getLogger(__name__)


def z_diagonal(dim, dtype=float):
    r"""Return the diagonal for the operator :math:`Z^\otimes n`"""
    parity = np.zeros(dim, dtype=dtype)
    for i in range(dim):
        parity[i] = bin(i)[2:].count("1")
    return (-1) ** np.mod(parity, 2)


def expval_with_stddev(coeffs: np.ndarray, probs: np.ndarray, shots: int) -> Tuple[float, float]:
    """Compute expectation value and standard deviation.
    Args:
        coeffs: array of diagonal operator coefficients.
        probs: array of measurement probabilities.
        shots: total number of shots to obtain probabilities.
    Returns:
        tuple: (expval, stddev) expectation value and standard deviation.
    """
    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    sq_expval = (coeffs ** 2).dot(probs)
    variance = (sq_expval - expval ** 2) / shots

    # Compute standard deviation
    if variance < 0 and not np.isclose(variance, 0):
        logger.warning(
            "Encountered a negative variance in expectation value calculation."
            "(%f). Setting standard deviation of result to 0.",
            variance,
        )
    calc_stddev = np.sqrt(variance) if variance > 0 else 0.0
    return [expval, calc_stddev]


def stddev(probs, shots):
    """Calculate stddev dict"""
    ret = {}
    for key, prob in probs.items():
        std_err = np.sqrt(prob * (1 - prob) / shots)
        ret[key] = std_err
    return ret


def str2diag(string):
    """Transform diagonal from a string to a numpy array"""
    chars = {
        "I": np.array([1, 1], dtype=float),
        "Z": np.array([1, -1], dtype=float),
        "0": np.array([1, 0], dtype=float),
        "1": np.array([0, 1], dtype=float),
    }
    ret = np.array([1], dtype=float)
    for i in string:
        if i not in chars:
            raise QiskitError(f"Invalid diagonal string character {i}")
        ret = np.kron(chars[i], ret)
    return ret


def counts_probability_vector(
    counts: Counts,
    qubits: Optional[List[int]] = None,
    clbits: Optional[List[int]] = None,
    num_qubits: Optional[int] = None,
    return_shots: Optional[bool] = False,
) -> np.ndarray:
    """Compute a probability vector for all count outcomes.

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
    if clbits_len not in (0, qubits_len):
        raise QiskitError(
            "Num qubits ({}) does not match number of clbits ({}).".format(qubits_len, clbits_len)
        )

    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, clbits)

    # Get total number of qubits
    if num_qubits is None:
        num_qubits = len(next(iter(counts)))

    # Get vector
    vec = np.zeros(2 ** num_qubits, dtype=float)
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
        vec = np.reshape(vec, num_qubits * [2]).transpose(axes).reshape(vec.shape)
    if return_shots:
        return vec, shots
    return vec
