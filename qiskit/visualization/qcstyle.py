# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""mpl circuit visualization style."""

# Temporary import from 0.22.0 to be deprecated in future
# pylint: disable=unused-wildcard-import,wildcard-import
import warnings
from .circuit.qcstyle import *

warnings.warn(
    "The qiskit.visualization.qcstyle module is deprecated as of Qiskit 0.46.0 and will be removed "
    "for Qiskit 1.0. Use the qiskit.visualization.circuit.qcstyle as direct replacement.",
    stacklevel=2,
    category=DeprecationWarning,
)
