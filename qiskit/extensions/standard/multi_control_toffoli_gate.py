# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The multi-controlled Toffoli gate moved to qiskit/extensions/standard/x.py."""

import warnings
from .x import mcx as mct

warnings.warn('The multi_control_toffoli_gate module is deprecated as of 0.13.0 and will be '
              'removed no earlier than 3 months after this release date. The multi-controlled '
              'Toffoli and the ``mct`` function can be found in x.py along with the X gates.',
              DeprecationWarning, stacklevel=2)

__all__ = ['mct']
