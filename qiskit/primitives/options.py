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
Options for reference primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunOptions:
    """Runtime options for the reference implementation."""

    shots: int | None = None
    """
    The number of shots. If None, it calculates the probabilities.
    Otherwise, it samples from multinomial distributions.
    """
    seed: int | np.random.Generator | None = None
    """
    Set a fixed seed or generator for the multinomial distribution. If shots is None, this option
    is ignored.
    """


@dataclass
class ReferenceOptions:
    """Options for the primitive programs."""

    run_options: dict = field(default_factory=RunOptions)
    """Runtime options used in run method."""
