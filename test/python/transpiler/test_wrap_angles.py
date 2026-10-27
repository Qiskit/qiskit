# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Wrap angles pass testing"""

from test import QiskitTestCase

from math import pi
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Parameter, Qubit, IfElseOp
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import WrapAngles
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.transpiler import WrapAngleRegistry
from qiskit.circuit.library import RZXGate, RZZGate, XGate
from qiskit.converters import circuit_to_dag


class TestWrapAngles(QiskitTestCase):
    """Tests for WrapAngles pass."""

    def test_empty_dag(self):
        """Invalid callback"""
        circuit = QuantumCircuit(2)
        target = Target.from_configuration(["u"], 2, CouplingMap([[0, 1]]))

        def callback(_, __):
            raise NotADirectoryError("test_works")

        target.add_instruction(RZXGate(Parameter("x")), angle_bounds=[(0, 3.14)])
        registry = WrapAngleRegistry()
        registry.add_wrapper("rzx", callback)

        wrap_pass = WrapAngles(target, registry)
        circuit.rzx(6.28, 0, 1)
        with self.assertRaisesRegex(NotADirectoryError, "test_works"):
            wrap_pass(circuit)

    def test_combine_custom_gates(self):
        """Test custom gates are combined as presscribed."""

        class MyCustomGate(Gate):
            """A custom gate definition."""

            def __init__(self, angle):
                super().__init__("my_custom", 1, [angle])

        param = Parameter("a")
        circuit = QuantumCircuit(1)
        circuit.append(MyCustomGate(6.0), [0])
        target = Target(num_qubits=1)
        target.add_instruction(MyCustomGate(param), angle_bounds=[(0, 0.5)])

        def callback(angles, _qubits):
            angle = angles[0]
            if angle > 0:
                number_of_gates = angle / 0.5
            else:
                number_of_gates = (6.28 - angle) / 0.5
            dag = DAGCircuit()
            dag.add_qubits([Qubit()])
            for _ in range(int(number_of_gates)):
                dag.apply_operation_back(MyCustomGate(0.5), [dag.qubits[0]])
            return dag

        registry = WrapAngleRegistry()
        registry.add_wrapper("my_custom", callback)
        wrap_pass = WrapAngles(target, registry)
        res = wrap_pass(circuit)
        self.assertEqual(res.count_ops()["my_custom"], 12)

    def test_legacy_default_registy(self):
        """Test that the legacy Qiskit 2.2 import path works for the default registry."""
        from qiskit.transpiler.passes.utils.wrap_angles import WRAP_ANGLE_REGISTRY

        self.assertIs(WRAP_ANGLE_REGISTRY, WrapAngles.DEFAULT_REGISTRY)

    def test_control_flow(self):
        """Test that the WrapAngles class doesn't ignore RZZ angle bounds"""
        target = Target(num_qubits=2)
        target.add_instruction(RZZGate(Parameter("theta")), angle_bounds=[(0, pi / 2)])
        target.add_instruction(XGate())
        target.add_instruction(IfElseOp, name="if_else")

        def wrap_rzz(angles, _qubits):
            replacement = QuantumCircuit(2)
            replacement.x(0)
            replacement.rzz(-angles[0], 0, 1)
            replacement.x(0)
            return circuit_to_dag(replacement)

        WrapAngles.DEFAULT_REGISTRY.add_wrapper("rzz", wrap_rzz)

        qc = QuantumCircuit(2, 1)
        qc.rzz(-0.4, 0, 1)
        with qc.if_test((qc.clbits[0], 0)):
            qc.rzz(-0.4, 0, 1)

        out = transpile(qc, target=target, optimization_level=0)
        body = out.data[-1].operation.blocks[0]  # true branch of the if_else
        angle_top_level = [
            float(i.operation.params[0]) for i in out.data if i.operation.name == "rzz"
        ]
        angle_inside_if_else = [
            float(i.operation.params[0]) for i in body.data if i.operation.name == "rzz"
        ]
        self.assertEqual(angle_top_level, angle_inside_if_else)
