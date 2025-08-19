# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Enumeration of targeted basis types.
"""

import enum


class TargetBasisType(enum.Enum):
    """Type of the basis set targeted during transpilation.

    The default basis set :data:`DEFAULT` targets near-term devices, with transpilation
    optimized for reducing the 2-qubit gate count and/or the 2-qubit depth of the
    output circuit.

    The Clifford+T basis set :data:`CLIFFORD_T` targets fault-tolerant devices, with
    transpilation optimized for reducing the T-count and/or the T-depth of the output
    circuit.
    """

    DEFAULT = enum.auto()
    """The transpilation is optimized towards the default (near-term) basis set."""
    CLIFFORD_T = enum.auto()
    """The transpilation is optimized towards the Clifford+T basis set."""
