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

    def push_forward(self, forward_map):
        """"
        RENAME THIS FUNCTION
        permutation {a: b} replaced by {sigma(a): sigma(b)}
        for now both have same size, so can compute as: first apply
        sigma-inverse, then perm, then sigma
        """
        assert isinstance(forward_map, list)
        forward_map_inverse = _invert_permutation(forward_map)
        self.permutation = _compose_permutations(forward_map_inverse, self.permutation, forward_map)

    def copy(self):
        print(f"COPY")
        return FinalPermutation(self.permutation.copy())




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
