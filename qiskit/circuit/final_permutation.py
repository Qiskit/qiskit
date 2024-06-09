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

## ToDo: reimplement using PermutationGate ?
from qiskit.circuit import CircuitError


class FinalPermutation:
    """
    The notation is the same as for permutation gate, that is
    [which qubit gets mapped to location 0, which qubit gets mapped to location 1, ...]
    in this way circuit with implicit final permutation sigma can be replaced
    by circuit with PermutationGate([sigma])
    """
    def __init__(self, permutation=None):

        if permutation is None:
            permutation = []
        self.permutation = permutation

    def add_qubit(self):
        self.permutation.append(len(self.permutation))

    def num_qubits(self):
        return len(self.permutation)

    def is_identity(self):
        return all(from_index == to_index for from_index, to_index in enumerate(self.permutation))

    def to_identity(self):
        for from_index in range(len(self.permutation)):
            self.permutation[from_index] = from_index

    def compose(self, other, front=True):
        """front=True: apply self, then other; front=False: apply other, then self."""
        if front:
            self.permutation = _compose_permutations(self.permutation, other)
        else:
            self.permutation = _compose_permutations(other, self.permutation)

    def __repr__(self):
        return str(self.permutation)

    def copy(self):
        print(f"COPY")
        return FinalPermutation(self.permutation.copy())

    def push_using_mapping(self, forward_map, num_target_qubits=None) -> "FinalPermutation":
        r"""
        Given a permutation :math:`\sigma: V \rightarrow V`, and a map
        :math:`\tau: V \rightarrow P`, "pushes" :math:`\sigma` to a
        permutation :math:`\tilde{\sigma}: P\rightarrow P`.

        ToDo: include proper definition of sigma-tilde.

        To do: Explain num target qubits
        """

        if num_target_qubits is None:
            num_target_qubits = len(forward_map)

        target_permutation = list(range(num_target_qubits))

        if isinstance(forward_map, list):
            for inp, out in enumerate(forward_map):
                target_permutation[out] = forward_map[self.permutation[inp]]
        elif isinstance(forward_map, dict):
            for inp, out in forward_map.items():
                target_permutation[out] = forward_map[self.permutation[inp]]
        else:
            raise CircuitError("The map should be given either as a list or as a dict.")



        return FinalPermutation(target_permutation)


def _invert_permutation(perm):
    assert isinstance(perm, list)

    """Finds inverse of a permutation."""
    inverse_map = {inp: out for out, inp in enumerate(perm)}
    return [inverse_map[inp] for inp in range(len(perm))]


def _compose_permutations(*perms):
    """Compose multiple permutations, with the permutations applied in the
    order they appear in the list.
    """
    out = range(len(perms[0]))
    for perm in perms:
        out = [perm[i] for i in out]
    return out
