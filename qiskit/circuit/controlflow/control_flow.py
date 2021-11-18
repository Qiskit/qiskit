# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"Container to encapsulate all control flow operations."

from abc import ABC, abstractmethod
from typing import Tuple

from qiskit.circuit import QuantumCircuit, Instruction


class ControlFlowOp(Instruction, ABC):
    """Abstract class to encapsulate all control flow operations."""

    @property
    @abstractmethod
    def blocks(self) -> Tuple[QuantumCircuit, ...]:
        """Tuple of QuantumCircuits which may be executed as part of the
        execution of this ControlFlowOp. May be parameterized by a loop
        parameter to be resolved at run time.
        """
        pass
