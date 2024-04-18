# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Utilities to invert and compose permutations.

A permutation :math:`\sigma` is represented as a list of integers, with the integer
at position ``i`` corresponding to :math:`\sigma(i)`. As an example, ``[2, 4, 3, 0, 1]``
represents the permutation that maps ``0`` to ``2``, ``1`` to ``4``, ``2`` to ``3``,
``3`` to ``0`` and ``4`` to ``1``.

This notation is the same as used in routing passes. However, it's the inverse of the
notation used in ``PermutationGate``.
"""


def _invert_permutation(perm):
    """Finds inverse of a permutation."""
    inverse_map = {inp: out for out, inp in enumerate(perm)}
    return [inverse_map[inp] for inp in range(len(perm))]


def _compose_permutations(*perms):
    """Compose multiple permutations, with the permutations applied in the
    order they appear in the list. For convenience, we allow some permutations
    to be ``None`` which represents identity permutations.
    """
    perms = [perm for perm in perms if perm is not None]
    if not perms:
        return None
    out = range(len(perms[0]))
    for perm in perms:
        out = [perm[i] for i in out]
    return out
