# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-member,invalid-name,redefined-outer-name,missing-docstring

"""
Directed graph object for representing coupling between physical qubits.

DEPRECATED IN TERRA 0.8+
"""
import warnings


def CouplingMap(couplinglist=None):
    warnings.warn('qiskit.mapper.CouplingMap has moved to '
                  'qiskit.transpiler.CouplingMap.', DeprecationWarning)
    from qiskit.transpiler.coupling import CouplingMap
    return CouplingMap(couplinglist)
