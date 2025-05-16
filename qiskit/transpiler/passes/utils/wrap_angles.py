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

"""Wrap angles pass for respecting target angle bounds."""

from qiskit.transpiler.basepasses import TransformationPass

from qiskit._accelerate import wrap_angles


class WrapAngles(TransformationPass):
    """Wrap angles outside the bound specified in the target.

    This pass will check all the gates in the circuit and check if there are any gates outside the
    bound specified in the target. If any gates outside the bound are identified the callback in
    the target will be called to substitute the gate outside the bound with an equivalent subcircuit.
    This pass does not run on gates that are parameterized, even if the gate has unparameterized
    parameters outside a specified bound. If there are parameterized gates they must be bound for
    the angles to be treated as inside the bounds
    """

    def __init__(self, target):
        super().__init__()
        self.target = target

    def run(self, dag):
        wrap_angles.wrap_angles(dag, self.target)
        return dag
