# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for monitoring jobs, backends, etc."""

import warnings

from .job_monitor import job_monitor
from .overview import backend_monitor, backend_overview


warnings.warn(
    "qiskit.tools.monitor is deprecated and will be removed in Qiskit 1.0.0", DeprecationWarning, 2
)
