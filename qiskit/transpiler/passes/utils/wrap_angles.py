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
from qiskit._accelerate.angle_bound_registry import WrapAngleRegistry

WRAP_ANGLE_REGISTRY = WrapAngleRegistry()


class WrapAngles(TransformationPass):
    """Wrap angles outside the bound specified in the target.

    This pass will check all the gates in the circuit and check if there are any gates outside the
    bound specified in the target. If any gates outside the bound are identified, the callback in
    the target will be called to substitute the gate outside the bound with an equivalent subcircuit.
    This pass does not run on gates that are parameterized, even if the gate has unparameterized
    parameters outside a specified bound. If there are parameterized gates in the circuit they will
    be ignored by this pass as bound angles are necessary to transform the gate. For example the below
    example demonstrates how the callback mechanism and registration works, but doesn't show a useful
    transformation, but is simple to follow:

    .. plot::
       :alt: Circuit digram of the output from running the WrapAngles pass
       :include-source:

       from qiskit.circuit import Gate, Parameter, Qubit, QuantumCircuit
       from qiskit.circuit.library import RZGate
       from qiskit.dagcircuit import DAGCircuit
       from qiskit.transpiler.passes import WrapAngles
       from qiskit.transpiler.target import Target
       from qiskit.transpiler import WrapAngleRegistry

       param = Parameter("a")
       circuit = QuantumCircuit(1)
       circuit.rz(6.8, 0)
       target = Target(num_qubits=1)
       target.add_instruction(RZGate(param), angle_bounds=[(0, 0.5)])

       def callback(angles, _qubits):
           angle = angles[0]
           if angle > 0:
               number_of_gates = angle / 0.5
           else:
               number_of_gates = (6.28 - angle) / 0.5
           dag = DAGCircuit()
           dag.add_qubits([Qubit()])
           for _ in range(int(number_of_gates)):
               dag.apply_operation_back(RZGate(0.5), [dag.qubits[0]])
           return dag

       registry = WrapAngleRegistry()
       registry.add_wrapper("rz", callback)
       wrap_pass = WrapAngles(target, registry)
       res = wrap_pass(circuit)
       res.draw("mpl")

    Args:
        target(Target): The :class:`.Target` that
        registry(WrapAngleRegistry): The registry of wrapping functions used
            by the pass to wrap the angles of a gate. If not specified the
            global ``WRAP_ANGLE_REGISTRY`` object defined in this module will be used.
            Unless you are planning to run this pass standalone or are building a
            custom :class:`~.transpiler.PassManager` including this pass you will want
            to rely on ``WRAP_ANGLE_REGISTRY``.
    """

    def __init__(self, target, registry=None):
        super().__init__()
        self.target = target
        if registry:
            self.registry = registry
        else:
            self.registry = WRAP_ANGLE_REGISTRY

    def run(self, dag):
        wrap_angles.wrap_angles(dag, self.target, self.registry)
        return dag
