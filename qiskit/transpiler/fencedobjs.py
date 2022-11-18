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

""" Fenced objects are wraps for raising TranspilerError when they are modified."""

from qiskit.passmanager.fencedobjs import FencedObject


class FencedDAGCircuit(FencedObject):
    """A dag circuit that cannot be modified (via remove_op_node)"""

    # FIXME: add more fenced methods of the dag after dagcircuit rewrite
    def __init__(self, dag_circuit_instance):
        super().__init__(dag_circuit_instance, ["remove_op_node"])
