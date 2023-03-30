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

"""Rearrange the direction of the cx nodes to match the directed coupling map."""

import warnings
from qiskit.transpiler.passes.utils.gate_direction import GateDirection


class CXDirection(GateDirection):
    """Deprecated: use :class:`qiskit.transpiler.passes.GateDirection` pass instead."""

    def __init__(self, coupling_map):
        super().__init__(coupling_map)
        warnings.warn(
            "The CXDirection pass has been deprecated "
            "and replaced by a more generic GateDirection pass.",
            DeprecationWarning,
            stacklevel=2,
        )
