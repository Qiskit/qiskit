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
Fidelity result class
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class StateFidelityResult:
    """Result of Fidelity computation.

    Args:
        values (list[float]): List of fidelity values for each pair of input circuits.
        metadata: Additional information on the fidelity calculations.
    """

    values: list[float]
    metadata: list[dict[str, Any]]
