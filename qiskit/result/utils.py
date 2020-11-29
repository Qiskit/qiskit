# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for working with Result counts."""

from collections import Counter

from qiskit.exceptions import QiskitError


def marginal_counts(result, indices=None, format_marginal=False):
    """Marginalize counts from an experiment over some indices of interest.

    Args:
        result (dict): result to be marginalized as a dict of counts.
        indices (list(int) or None): The bit positions of interest
            to marginalize over. If ``None`` (default), do not marginalize at all.
        format_marginal (bool): Default: False. If True, takes the output of
            marginalize and formats it with placeholders between cregs and
            for non-indices.

    Returns:
        dict(str, int): a dictionary with the observed counts, marginalized to
        only account for frequency of observations of bits of interest.

    Raises:
        QiskitError: in case of invalid indices to marginalize over.
    """
    counts = _marginalize(result, indices)
    if format_marginal and indices is not None:
        counts = _format_marginal(result, counts, indices)
    return counts


def _marginalize(counts, indices=None):
    # Extract total number of clbits from first count key
    # We trim the whitespace separating classical registers
    # and count the number of digits
    num_clbits = len(next(iter(counts)).replace(' ', ''))

    # Check if we do not need to marginalize. In this case we just trim
    # whitespace from count keys
    if (indices is None) or set(range(num_clbits)) == set(indices):
        ret = {}
        for key, val in counts.items():
            key = key.replace(' ', '')
            ret[key] = val
        return ret

    if not set(indices).issubset(set(range(num_clbits))):
        raise QiskitError('indices must be in range [0, {}].'.format(num_clbits-1))

    # Sort the indices to keep in decending order
    # Since bitstrings have qubit-0 as least significant bit
    indices = sorted(indices, reverse=True)

    # Build the return list
    new_counts = Counter({})
    for key, val in counts.items():
        new_key = ''.join([key.replace(' ', '')[-idx-1] for idx in indices])
        new_counts[new_key] += val
    return dict(new_counts)


def _format_marginal(result, counts, indices):
    """Take the output of marginalize and add placeholders for
    multiple cregs and non-indices."""
    format_counts = {}
    res_template = next(iter(result))
    res_len = len(res_template.replace(' ', ''))
    indices_rev = sorted(indices, reverse=True)

    for count in counts:
        index_dict = dict(zip(indices_rev, count))
        res_bits = ''.join([index_dict[index] if index in index_dict else 'x'
                            for index in range(res_len)])[::-1]
        for index, bit in enumerate(res_template):
            if bit == ' ':
                res_bits = res_bits[:index] + ' ' + res_bits[index:]
        format_counts[res_bits] = counts[count]
    return format_counts
