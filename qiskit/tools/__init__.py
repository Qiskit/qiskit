# -*- coding: utf-8 -*-

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

"""Helper module for simplified Qiskit usage.

The functions in this module provide convenience helpers for accessing commonly
used features of the SDK in a simplified way. They support a small subset of
scenarios and flows: for more advanced usage, it is encouraged to instead
refer to the documentation of each component and use them separately.
"""

from .parallel import parallel_map
