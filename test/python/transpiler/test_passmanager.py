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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Test the passmanager logic"""

import copy

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import U2Gate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, PropertySet, TransformationPass, FlowController
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import Optimize1qGates, Unroller
from qiskit.test import QiskitTestCase


class TestPassManager(QiskitTestCase):
    """Test Pass manager logic."""

    def test_callback(self):
        """Test the callback parameter."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start_dag = circuit_to_dag(expected_start)

        expected_end = QuantumCircuit(qr)
        expected_end.append(U2Gate(0, np.pi), [qr[0]])
        expected_end_dag = circuit_to_dag(expected_end)

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict["dag"] = copy.deepcopy(kwargs["dag"])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(Unroller(["u2"]))
        passmanager.append(Optimize1qGates())
        passmanager.run(circuit, callback=callback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(calls[0]["count"], 0)
        self.assertEqual(calls[0]["pass_"].name(), "Unroller")
        self.assertEqual(expected_start_dag, calls[0]["dag"])
        self.assertIsInstance(calls[0]["time"], float)
        self.assertEqual(calls[0]["property_set"], PropertySet())
        self.assertEqual("MyCircuit", calls[0]["dag"].name)
        self.assertEqual(len(calls[1]), 5)
        self.assertEqual(calls[1]["count"], 1)
        self.assertEqual(calls[1]["pass_"].name(), "Optimize1qGates")
        self.assertEqual(expected_end_dag, calls[1]["dag"])
        self.assertIsInstance(calls[0]["time"], float)
        self.assertEqual(calls[0]["property_set"], PropertySet())
        self.assertEqual("MyCircuit", calls[1]["dag"].name)

    def test_callback_with_pass_requires(self):
        """Test the callback with a pass with another pass requirement."""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        circuit.z(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.z(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.z(qr[0])
        expected_start.cx(qr[0], qr[2])
        expected_start.z(qr[0])
        expected_start_dag = circuit_to_dag(expected_start)

        expected_end = QuantumCircuit(qr)
        expected_end.cx(qr[0], qr[2])
        expected_end_dag = circuit_to_dag(expected_end)

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict["dag"] = copy.deepcopy(kwargs["dag"])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(basis_gates=["u1", "u2", "u3", "cx"]))
        passmanager.run(circuit, callback=callback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(calls[0]["count"], 0)
        self.assertEqual(calls[0]["pass_"].name(), "CommutationAnalysis")
        self.assertEqual(expected_start_dag, calls[0]["dag"])
        self.assertIsInstance(calls[0]["time"], float)
        self.assertIsInstance(calls[0]["property_set"], PropertySet)
        self.assertEqual("MyCircuit", calls[0]["dag"].name)
        self.assertEqual(len(calls[1]), 5)
        self.assertEqual(calls[1]["count"], 1)
        self.assertEqual(calls[1]["pass_"].name(), "CommutativeCancellation")
        self.assertEqual(expected_end_dag, calls[1]["dag"])
        self.assertIsInstance(calls[0]["time"], float)
        self.assertIsInstance(calls[0]["property_set"], PropertySet)
        self.assertEqual("MyCircuit", calls[1]["dag"].name)

    def test_to_flow_controller(self):
        """Test that conversion to a `FlowController` works, and the result can be added to a
        circuit and conditioned, with the condition only being called once."""

        class DummyPass(TransformationPass):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def run(self, dag):
                return dag

        def repeat(count):  # pylint: disable=unused-argument
            def condition(_):
                nonlocal count
                if not count:
                    return False
                count -= 1
                return True

            return condition

        def make_inner(prefix):
            inner = PassManager()
            inner.append(DummyPass(f"{prefix} 1"))
            inner.append(DummyPass(f"{prefix} 2"), condition=lambda _: False)
            inner.append(DummyPass(f"{prefix} 3"), condition=lambda _: True)
            inner.append(DummyPass(f"{prefix} 4"), do_while=repeat(1))
            return inner.to_flow_controller()

        self.assertIsInstance(make_inner("test"), FlowController)

        outer = PassManager()
        outer.append(make_inner("first"))
        outer.append(make_inner("second"), condition=lambda _: False)
        # The intent of this `condition=repeat(1)` is to ensure that the outer condition is only
        # checked once and not flattened into the inner controllers; an inner pass invalidating the
        # condition should not affect subsequent passes once the initial condition was met.
        outer.append(make_inner("third"), condition=repeat(1))

        calls = []

        def callback(pass_, **_):
            self.assertIsInstance(pass_, DummyPass)
            calls.append(pass_.x)

        outer.run(QuantumCircuit(), callback=callback)

        expected = [
            "first 1",
            "first 3",
            # it's a do-while loop, not a while, which is why the `repeat(1)` gives two calls
            "first 4",
            "first 4",
            # If the outer pass-manager condition is called more than once, then only the first of
            # the `third` passes will appear.
            "third 1",
            "third 3",
            "third 4",
            "third 4",
        ]
        self.assertEqual(calls, expected)
