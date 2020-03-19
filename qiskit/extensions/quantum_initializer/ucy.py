# -*- coding: utf-8 -*-

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

"""The uniformly controlled RY gate.

This module is deprecated, see ucry.py
"""

import warnings
# pylint: disable=unused-import
from qiskit.extensions.quantum_initializer.ucry import UCY, ucy

warnings.warn('This module is deprecated. The UCRYGate/UCY can now be found in ucry.py',
              category=DeprecationWarning, stacklevel=2)
