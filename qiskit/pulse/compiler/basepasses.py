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

"""A base pass for Qiskit PulseIR compilation."""

import warnings
from abc import ABC, abstractmethod

from qiskit.passmanager.base_tasks import GenericPass
from qiskit.transpiler.target import Target
from qiskit.pulse.ir import SequenceIR


class TransformationPass(GenericPass, ABC):
    """A base transform pass for Qiskit PulseIR.

    A transform pass modifies the input Qiskit PulseIR and returns an updated PulseIR.
    The returned object can be new instance, or the pass can mutate and return the same object.
    """

    def __init__(
        self,
        target: Target,
    ):
        """Create new transform pass.

        Args:
            target: System configuration information presented in the form of Qiskit model.
        """
        super().__init__()
        self.target = target

    @abstractmethod
    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:
        pass

    def __hash__(self) -> int:
        warnings.warn(
            f"{self.__class__} does not explicitly define a protocol to calculate hash. "
            "This pass created the hash only by considering the class name. "
            "Qiskit flow controller assumes passes with the identical hash are idempotent, "
            "and it may skip execution of the other passes in the pipeline.",
            RuntimeWarning,
        )
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        warnings.warn(
            f"{self.__class__} does not explicitly define a protocol to evaluate equality. "
            "Two pass objects instantiated individually with the same configuration may be "
            "considered as different passes.",
            RuntimeWarning,
        )
        super().__eq__(other)


class AnalysisPass(GenericPass, ABC):
    """A base analysis pass for Qiskit PulseIR.

    An analysis pass performs investigation on the input Qiskit PulseIR.
    The information obtained may be stored in the property set.
    This pass returns nothing.
    """

    def __init__(
        self,
        target: Target,
    ):
        """Create new transform pass.

        Args:
            target: System configuration information presented in the form of Qiskit model.
        """
        super().__init__()
        self.target = target

    @abstractmethod
    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> None:
        pass

    def __hash__(self) -> int:
        warnings.warn(
            f"{self.__class__} does not explicitly define a protocol to calculate hash. "
            "This pass created the hash only by considering the class name. "
            "Qiskit flow controller assumes passes with the identical hash are idempotent, "
            "and it may skip execution of the other passes in the pipeline.",
            RuntimeWarning,
        )
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        warnings.warn(
            f"{self.__class__} does not explicitly define a protocol to evaluate equality. "
            "Two pass objects instantiated individually with the same configuration may be "
            "considered as different passes.",
            RuntimeWarning,
        )
        super().__eq__(other)
