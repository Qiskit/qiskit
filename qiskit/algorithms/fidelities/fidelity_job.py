# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Sampler result class
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.providers import JobStatus


@dataclass(frozen=True)
class FidelityJob:
    """Result of Fidelity class.
    Args:
        results: np.array with calculated state overlaps
        status: JobStatus for the Fidelity job
    """

    result: np.array
    status: JobStatus
