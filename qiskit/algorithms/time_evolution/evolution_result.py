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
from typing import Dict, Any, Optional

from qiskit.opflow import OperatorBase


class EvolutionResult:
    """Class for holding evolution result and relevant metadata."""

    def __init__(self, evolved_object: OperatorBase, metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            evolved_object: An evolved quantum state or an evolved quantum observable.
            metadata: A dictionary with algorithm-specific metadata. Keys contain strings that name
                data stores as a corresponding value.
        """
        self._evolved_object = evolved_object
        self._metadata = metadata

    @property
    def evolved_object(self):
        """Returns an evolved quantum state or an evolved quantum observable."""
        return self._evolved_object

    @property
    def metadata(self):
        """Returns a dictionary with algorithm-specific metadata."""
        return self._metadata
