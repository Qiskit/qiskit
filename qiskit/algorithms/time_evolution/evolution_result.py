# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Class for holding evolution result and relevant metadata."""
from typing import Optional

from qiskit.algorithms import AlgorithmResult
from qiskit.opflow import OperatorBase


class EvolutionResult(AlgorithmResult):
    """Class for holding evolution result."""

    def __init__(
        self,
        evolved_state: Optional[OperatorBase] = None,
        evolved_observable: Optional[OperatorBase] = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state; mutually exclusive with evolved_observable.
            evolved_observable: An evolved quantum observable; mutually exclusive with
                evolved_state.

        Raises:
            ValueError: If both or none evolved_state and evolved_observable are provided.
        """

        if evolved_state is not None and evolved_observable is not None:
            raise ValueError(
                "evolved_state and evolved_observable are mutually exclusive; both provided."
            )

        if evolved_state is None and evolved_observable is None:
            raise ValueError(
                "One of evolved_state or evolved_observable must be provided; none provided."
            )

        self.evolved_state = evolved_state
        self.evolved_observable = evolved_observable
