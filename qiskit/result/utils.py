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

"""Utility functions for working with Results."""

from collections import Counter
from copy import deepcopy

from qiskit.exceptions import QiskitError
from qiskit.result.result import Result
from qiskit.result.postprocess import _bin_to_hex


def marginal_counts(result, indices=None, inplace=False, format_marginal=False):
    """Marginalize counts from an experiment over some indices of interest.

    Args:
        result (dict or Result): result to be marginalized
            (a Result object or a dict(str, int) of counts).
        indices (list(int) or None): The bit positions of interest
            to marginalize over. If ``None`` (default), do not marginalize at all.
        inplace (bool): Default: False. Operates on the original Result
            argument if True, leading to loss of original Job Result.
            It has no effect if ``result`` is a dict.
        format_marginal (bool): Default: False. If True, takes the output of
            marginalize and formats it with placeholders between cregs and
            for non-indices.

    Returns:
        Result or dict(str, int): A Result object or a dictionary with
            the observed counts, marginalized to only account for frequency
            of observations of bits of interest.

    Raises:
        QiskitError: in case of invalid indices to marginalize over.
    """
    if isinstance(result, Result):
        if not inplace:
            result = deepcopy(result)
        for i, experiment_result in enumerate(result.results):
            counts = result.get_counts(i)
            new_counts = _marginalize(counts, indices)
            new_counts_hex = {}
            for k, v in new_counts.items():
                new_counts_hex[_bin_to_hex(k)] = v
            experiment_result.data.counts = new_counts_hex
            experiment_result.header.memory_slots = len(indices)
            csize = experiment_result.header.creg_sizes
            experiment_result.header.creg_sizes = _adjust_creg_sizes(csize, indices)
        return result
    else:
        marg_counts = _marginalize(result, indices)
        if format_marginal and indices is not None:
            marg_counts = _format_marginal(result, marg_counts, indices)
        return marg_counts


def _adjust_creg_sizes(creg_sizes, indices):
    """Helper to reduce creg_sizes to match indices"""

    # Zero out creg_sizes list
    new_creg_sizes = [[creg[0], 0] for creg in creg_sizes]
    indices_sort = sorted(indices)

    # Get creg num values and then convert to the cumulative last index per creg.
    # e.g. [2, 1, 3] => [1, 2, 5]
    creg_nums = [x for _, x in creg_sizes]
    creg_limits = [sum(creg_nums[0:x:1]) - 1 for x in range(0, len(creg_nums) + 1)][1:]

    # Now iterate over indices and find which creg that index is in.
    # When found increment the creg size
    creg_idx = 0
    for ind in indices_sort:
        for idx in range(creg_idx, len(creg_limits)):
            if ind <= creg_limits[idx]:
                creg_idx = idx
                new_creg_sizes[idx][1] += 1
                break
    # Throw away any cregs with 0 size
    new_creg_sizes = [creg for creg in new_creg_sizes if creg[1] != 0]
    return new_creg_sizes


def _marginalize(counts, indices=None):
    """Get the marginal counts for the given set of indices"""
    num_clbits = len(next(iter(counts)).replace(" ", ""))

    # Check if we do not need to marginalize and if so, trim
    # whitespace and '_' and return
    if (indices is None) or set(range(num_clbits)) == set(indices):
        ret = {}
        for key, val in counts.items():
            key = _remove_space_underscore(key)
            ret[key] = val
        return ret

    if not indices or not set(indices).issubset(set(range(num_clbits))):
        raise QiskitError(f"indices must be in range [0, {num_clbits - 1}].")

    # Sort the indices to keep in descending order
    # Since bitstrings have qubit-0 as least significant bit
    indices = sorted(indices, reverse=True)

    # Build the return list
    new_counts = Counter()
    for key, val in counts.items():
        new_key = "".join([_remove_space_underscore(key)[-idx - 1] for idx in indices])
        new_counts[new_key] += val
    return dict(new_counts)


def _format_marginal(counts, marg_counts, indices):
    """Take the output of marginalize and add placeholders for
    multiple cregs and non-indices."""
    format_counts = {}
    counts_template = next(iter(counts))
    counts_len = len(counts_template.replace(" ", ""))
    indices_rev = sorted(indices, reverse=True)

    for count in marg_counts:
        index_dict = dict(zip(indices_rev, count))
        count_bits = "".join(
            [index_dict[index] if index in index_dict else "_" for index in range(counts_len)]
        )[::-1]
        for index, bit in enumerate(counts_template):
            if bit == " ":
                count_bits = count_bits[:index] + " " + count_bits[index:]
        format_counts[count_bits] = marg_counts[count]
    return format_counts


def _remove_space_underscore(bitstring):
    """Removes all spaces and underscores from bitstring"""
    return bitstring.replace(" ", "").replace("_", "")
