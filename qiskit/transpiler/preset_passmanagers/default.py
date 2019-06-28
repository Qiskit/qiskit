# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
