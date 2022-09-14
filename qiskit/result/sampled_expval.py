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

"""Routines for computing expectation values from sampled distributions"""
import numpy as np

from qiskit.exceptions import QiskitError
from .distributions import QuasiDistribution, ProbDistribution


# A dict defining the diagonal of each non-identity operator
OPERS = {'Z': [1, -1], '0': [1, 0], '1':[0, 1]}


def sampled_expectation_value(dist, oper):
    """Computes expectation value from a sampled distribution

    Parameters:
        dist (Counts or QuasiDistribution or ProbDistribution or dict): Input sampled distribution
        oper (str or Pauli or PauliOp or PauliSumOp)
    """
    from .counts import Counts
    from qiskit.quantum_info import Pauli, SparsePauliOp
    from qiskit.opflow import PauliOp, PauliSumOp
    # This should be removed when these return bit-string keys
    if isinstance(dist, (QuasiDistribution, ProbDistribution)):
        dist = dist.binary_probabilities()
    
    if not isinstance(dist, (Counts, QuasiDistribution, ProbDistribution, dict)):
        raise QiskitError('Invalid input distribution type')
    if isinstance(oper, str):
        oper_strs = [oper.upper()]
        coeffs = [1.0]
    elif isinstance(oper, Pauli):
        oper_strs = [oper.to_label()]
        coeffs = [1.0]
    elif isinstance(oper, PauliOp):
        oper_strs = [oper.primitive.to_label()]
        coeffs = [1.0]
    elif isinstance(oper, PauliSumOp):
        _lst = oper.primitive.to_list()
        oper_strs = [item[0] for item in _lst]
        coeffs = [item[1] for item in _lst]
    elif isinstance(oper, SparsePauliOp):
        oper_strs = oper.paulis.to_labels()
        coeffs = oper.coeffs
    else:
        raise QiskitError('Invalid operator type')
    out = 0
    for idx, string in enumerate(oper_strs):
        out += coeffs[idx]*_bitstring_expval(dist, string)

    return out.real


def _bitstring_expval(dist, oper_str):
    """Computes the expectation value of a sampled distribution
    for a operator expressed by a string.

    Parameters:
        dist (Counts or QuasiDistribution or ProbDistribution or dict): A a sampled distribution
        oper_str (str): An operator string

    Returns:
        float: Expectation value
    """
    # Sparsify operator string to just non-identity terms
    str_len = len(oper_str)
    inds = np.array([kk for kk in range(str_len) if oper_str[kk] != 'I'], np.int32)
    short_str = ''.join(oper_str[kk] for kk in inds)
    # Compute denominator
    denom = sum(dist.values())
    # Do the actual expval here
    exp_val = 0
    for bits, val in dist.items():
        temp_prod = 1.0
        for idx, oper in enumerate(short_str):
            temp_prod *= OPERS[oper][int(bits[inds[idx]])]
        exp_val += val * temp_prod
    exp_val /= denom
    return exp_val
