#!/usr/bin/env python3

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

"""Report some details of the python environment, for debugging"""

import numpy as np

np.show_config()

try:  # only available starting numpy 1.24.0
    np.show_runtime()
except AttributeError:
    pass
