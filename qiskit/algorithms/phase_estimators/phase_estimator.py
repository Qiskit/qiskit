# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Phase Estimator interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.algorithm_result import AlgorithmResult


class PhaseEstimator(ABC):
    """The Phase Estimator interface.

    Algorithms that can compute a phase for a unitary operator and initial state may implement this
    interface to allow different algorithms to be used interchangeably.

    The phase returned is a canonical phase determined by the specific algorithm, such as the most
    likely phase. In addition, the algorithm may provide an interface to retrieve phases by other
    criteria.
    """

    @abstractmethod
    def estimate(
        self,
        unitary: QuantumCircuit,
        state_preparation: QuantumCircuit | None = None,
    ) -> "PhaseEstimatorResult":
        """Estimate the phase."""
        raise NotImplementedError

    @staticmethod
    def _get_reversed_bitstring(length: int, number: int) -> str:
        return f"{number:b}".zfill(length)[::-1]


class PhaseEstimatorResult(AlgorithmResult):
    """Phase Estimator Result."""

    @property
    @abstractmethod
    def phase(self) -> float:
        r"""Return the estimated phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. In case the phase estimation algorithm
        computes more than one phase, this attribute returns a canonical single phase; for
        example, the most likely phase.
        """
        raise NotImplementedError
