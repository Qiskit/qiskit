# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Phase Estimator interface."""

from typing import Optional
from abc import ABC, abstractmethod, abstractproperty
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.algorithm_result import AlgorithmResult


class PhaseEstimator(ABC):
    """The Phase Estimator interface.

    Algorithms that can compute a phase for a unitary operator and
    initial state may implement this interface to allow different
    algorithms to be used interchangeably.
    """

    @abstractmethod
    def estimate(self,
                 unitary: Optional[QuantumCircuit] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 pe_circuit: Optional[QuantumCircuit] = None,
                 num_unitary_qubits: Optional[int] = None) -> 'PhaseEstimatorResult':
        """Estimate the phase."""
        raise NotImplementedError


class PhaseEstimatorResult(AlgorithmResult):
    """Phase Estimator Result."""

    @abstractproperty
    def most_likely_phase(self) -> float:
        r"""Return the estimated phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. It is assumed that the input vector is an
        eigenvector of the unitary so that the peak of the probability density occurs at the bit
        string that most closely approximates the true phase.
        """
        raise NotImplementedError
