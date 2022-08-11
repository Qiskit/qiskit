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
"""Run options for Primitives."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunOptions:
    ...


@dataclass
class Settings:
    run_options: RunOptions


@dataclass
class ReferenceRunOptions(RunOptions):
    shots: int | None = None
    seed: int | np.random.Generator | None = None


@dataclass
class ReferenceSettings(Settings):
    run_options: RunOptions = field(default_factory=ReferenceRunOptions)
