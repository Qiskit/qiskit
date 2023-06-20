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

"""Check if the CNOTs follow the right direction with respect to the coupling map.."""

from qiskit.transpiler.passes.utils.check_gate_direction import CheckGateDirection
from qiskit.utils.deprecation import deprecate_func


class CheckCXDirection(CheckGateDirection):
    """Deprecated: use :class:`qiskit.transpiler.passes.CheckGateDirection` pass instead."""

    @deprecate_func(
        additional_msg="Instead, use the more generic :class:`~.CheckGateDirection` pass.",
        since="0.21.0",
    )
    def __init__(self, coupling_map=None, target=None):
        super().__init__(coupling_map=coupling_map, target=target)
