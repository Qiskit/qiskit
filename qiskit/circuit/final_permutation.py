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

    def __init__(self):
        # for now: original index -> mapped to index
        self.permutation = []

    def add_qubit(self):
        self.permutation.append(len(self.permutation))

    def num_qubits(self):
        return len(self.permutation)

    def is_identity(self):
        return all(from_index == to_index for from_index, to_index in enumerate(self.permutation))

    def to_identity(self):
        for from_index in range(len(self.permutation)):
            self.permutation[from_index] = from_index
