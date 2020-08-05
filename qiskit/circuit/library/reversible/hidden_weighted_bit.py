# -*- coding: utf-8 -*-

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

"""Hidden weighted bit circuits."""

from qiskit.circuit import QuantumCircuit

class HiddenWeightedBit4(QuantumCircuit):
    def __init__(self) -> None:
        """Create Hidden weighted bit circuit for 4 qubits.
        """

        super().__init__(4, name="Hidden Weighted Bit 4")
        self.cx(0, 1)
        self.cx(3, 1)
        self.cx(1, 2)
        self.cx(2, 0)
        self.cx(0, 1)
        self.ccx(0, 1, 3)
        self.ccx(2, 3, 0)
        self.cx(1, 2)
        self.cx(2, 0)
        self.ccx(0, 2, 3)
        self.cx(3, 1)
        self.cx(0, 2)

class HiddenWeightedBit5(QuantumCircuit):
    def __init__(self) -> None:
        """Create Hidden weighted bit circuit for 5 qubits.
        """

        super().__init__(5, name="Hidden Weighted Bit 5")
        self.mcx([3], 4)
        self.mcx([0], 2)
        self.mcx([1, 2], 0)
        self.mcx([0], 3)
        self.mcx([3], 0)
        self.mcx([4], 3)
        self.mcx([3, 4], 0)
        self.mcx([0], 2)
        self.mcx([2, 4], 3)
        self.mcx([4], 2)
        self.mcx([3], 1)
        self.mcx([1], 2)
        self.mcx([], 1)
        self.mcx([2, 3, 4], 1)
        self.mcx([4], 2)
        self.mcx([3], 2)
        self.mcx([0], 4)
        self.mcx([2], 3)
        self.mcx([2, 3], 0)
        self.mcx([0, 2], 3)
        self.mcx([2], 0)
        self.mcx([4], 2)
        self.mcx([0, 1], 4)
        self.mcx([0], 1)
        self.mcx([1, 4], 0)
        self.mcx([0], 2)
        self.mcx([4], 2)
        self.mcx([0, 1, 4], 3)
        self.mcx([3], 2)
        self.mcx([], 1)
        self.mcx([3], 4)
        self.mcx([4], 3)
        self.mcx([2], 3)
        self.mcx([1], 3)
        self.mcx([3], 2)
        self.mcx([2], 1)
        self.mcx([0], 1)
        self.mcx([1], 0)