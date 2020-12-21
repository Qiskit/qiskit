# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Random Pauli circuit class."""

from typing import Optional
import numpy as np

from qiskit.circuit import QuantumCircuit


from .two_local import TwoLocal


class RandomPauli(TwoLocal):
    """The Random Pauli ansatz."""

    def __init__(self, num_qubits: Optional[int] = None, reps: int = 3,
                 seed: Optional[int] = None, insert_barriers: bool = False):
        # store a random number generator
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # store a dict to keep track of the random gates
        self._gates = dict()

        super().__init__(num_qubits, reps=reps, entanglement_blocks='cz', entanglement='pairwise',
                         insert_barriers=insert_barriers)

    def _invalidate(self):
        self._rng = np.random.default_rng(self._seed)  # reset number generator
        super()._invalidate()

    def _build_rotation_layer(self, param_iter, i):
        """Build a rotation layer."""
        layer = QuantumCircuit(*self.qregs)
        qubits = range(self.num_qubits)

        # if no gates for this layer were generated, generate them
        if i not in self._gates.keys():
            self._gates[i] = list(self._rng.choice(['rx', 'ry', 'rz'], self.num_qubits))
        # if not enough gates exist, add more
        elif len(self._gates[i]) < self.num_qubits:
            num_missing = self.num_qubits - len(self._gates[i])
            self._gates[i] += list(self._rng.choice(['rx', 'ry', 'rz'], num_missing))

        for j in qubits:
            getattr(layer, self._gates[i][j])(next(param_iter), j)

        # add the layer to the circuit
        self.compose(layer, inplace=True)

    @property
    def num_parameters_settable(self) -> int:
        """Return the number of settable parameters.

        Returns:
            The number of possibly distinct parameters.
        """
        return (self.reps + 1) * self.num_qubits

    def _build(self):
        super()._build()
        initial_layer = QuantumCircuit(self.num_qubits)
        initial_layer.ry(np.pi / 4, range(self.num_qubits))

        self.compose(initial_layer, front=True, inplace=True)
