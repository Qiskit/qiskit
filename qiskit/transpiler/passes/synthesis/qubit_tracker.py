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

"""A qubit state tracker for synthesizing operations with auxiliary qubits."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class QubitTracker:
    """Track qubits (by global index) and their state.

    The states are distinguished into clean (meaning in state :math:`|0\rangle`) or dirty (an
    unknown state).
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = [False] * num_qubits  # True: clean, False: dirty
        self.enabled = [True] * num_qubits  # True: allowed to use, False: not allowed to use
        self.ignored = [False] * num_qubits  # Internal scratch space

    def set_dirty(self, qubits):
        """Sets state of the given qubits to dirty."""
        for q in qubits:
            self.state[q] = False

    def set_clean(self, qubits):
        """Sets state of the given qubits to clean."""
        for q in qubits:
            self.state[q] = True

    def disable(self, qubits):
        """Disables using the given qubits."""
        for q in qubits:
            self.enabled[q] = False

    def enable(self, qubits):
        """Enables using the given qubits."""
        for q in qubits:
            self.enabled[q] = True

    def num_clean(self, ignored_qubits):
        """Returns the number of enabled clean qubits, ignoring the given qubits."""
        count = 0
        for q in ignored_qubits:
            self.ignored[q] = True
        for q in range(self.num_qubits):
            if (not self.ignored[q]) and self.enabled[q] and self.state[q]:
                count += 1
        for q in ignored_qubits:
            self.ignored[q] = False
        return count

    def num_dirty(self, ignored_qubits):
        """Returns the number of enabled dirty qubits, ignoring the given qubits."""
        count = 0
        for q in ignored_qubits:
            self.ignored[q] = True
        for q in range(self.num_qubits):
            if (not self.ignored[q]) and self.enabled[q] and not self.state[q]:
                count += 1
        for q in ignored_qubits:
            self.ignored[q] = False
        return count

    def borrow(self, num_qubits: int, ignored_qubits: Iterable[int] | None = None) -> list[int]:
        """Get ``num_qubits`` enabled qubits, excluding ``ignored_qubits`` and prioritizing
        clean qubits."""
        res = []
        for q in ignored_qubits:
            self.ignored[q] = True
        for q in range(self.num_qubits):
            if (not self.ignored[q]) and self.enabled[q] and self.state[q]:
                res.append(q)
        for q in range(self.num_qubits):
            if (not self.ignored[q]) and self.enabled[q] and not self.state[q]:
                res.append(q)
        for q in ignored_qubits:
            self.ignored[q] = False
        return res[:num_qubits]

    def copy(self) -> "QubitTracker":
        """Copies the qubit tracker."""
        tracker = QubitTracker(self.num_qubits)
        tracker.state = self.state.copy()
        tracker.enabled = self.enabled.copy()
        # no need to copy the scratch space (ignored)
        return tracker

    def replace_state(self, other: "QubitTracker", qubits):
        """Replaces the state of the given qubits by their state in the ``other`` tracker."""
        for q in qubits:
            self.state[q] = other.state[q]

    def __str__(self) -> str:
        """Pretty-prints qubit states."""
        s = ""
        for q in range(self.num_qubits):
            s += str(q) + ": "
            if not self.enabled[q]:
                s += "_"
            elif self.state[q]:
                s += "0"
            else:
                s += "*"
            s += "; "
        return s
