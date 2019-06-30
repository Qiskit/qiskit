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

"""A default passmanager."""

from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState


def default_pass_manager_simulator(transpile_config):
    """
    The default pass manager without a coupling map.

    Args:
        transpile_config (TranspileConfig)

    Returns:
        PassManager: A passmanager that just unrolls, without any optimization.
    """
    basis_gates = transpile_config.basis_gates

    pass_manager = PassManager()
    pass_manager.append(Unroller(basis_gates))
    pass_manager.append([RemoveResetInZeroState(), Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])

    return pass_manager
