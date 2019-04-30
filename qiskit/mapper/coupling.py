# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
