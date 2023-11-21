# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ApplyLayout pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers import common
from qiskit.providers.fake_provider import FakeVigoV2
from qiskit.transpiler import PassManager


class TestApplyLayout(QiskitTestCase):
    """Tests the ApplyLayout pass."""

    def test_trivial(self):
        """Test if the bell circuit with virtual qubits is transformed into
        the circuit with physical qubits under trivial layout.
        """
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        q = QuantumRegister(2, "q")
        expected = QuantumCircuit(q)
        expected.h(q[0])
        expected.cx(q[0], q[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 0, v[1]: 1})
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_raise_when_no_layout_is_supplied(self):
        """Test error is raised if no layout is found in property_set."""
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_raise_when_no_full_layout_is_given(self):
        """Test error is raised if no full layout is given."""
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 2, v[1]: 1})
        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_circuit_with_swap_gate(self):
        """Test if a virtual circuit with one swap gate is transformed into
        a circuit with physical qubits.

        [Circuit with virtual qubits]
          v0:--X---.---M(v0->c0)
               |   |
          v1:--X---|---M(v1->c1)
                   |
          v2:-----(+)--M(v2->c2)

         Initial layout: {v[0]: 2, v[1]: 1, v[2]: 0}

        [Circuit with physical qubits]
          q2:--X---.---M(q2->c0)
               |   |
          q1:--X---|---M(q1->c1)
                   |
          q0:-----(+)--M(q0->c2)
        """
        v = QuantumRegister(3, "v")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(v, cr)
        circuit.swap(v[0], v[1])
        circuit.cx(v[0], v[2])
        circuit.measure(v[0], cr[0])
        circuit.measure(v[1], cr[1])
        circuit.measure(v[2], cr[2])

        q = QuantumRegister(3, "q")
        expected = QuantumCircuit(q, cr)
        expected.swap(q[2], q[1])
        expected.cx(q[2], q[0])
        expected.measure(q[2], cr[0])
        expected.measure(q[1], cr[1])
        expected.measure(q[0], cr[2])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 2, v[1]: 1, v[2]: 0})
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_final_layout_is_updated(self):
        """Test that if vf2postlayout runs that we've updated the final layout."""
        qubits = 3
        qc = QuantumCircuit(qubits)
        for i in range(5):
            qc.cx(i % qubits, int(i + qubits / 2) % qubits)
        initial_pm = PassManager([SetLayout([1, 3, 4])])
        cmap = FakeVigoV2().coupling_map
        initial_pm += common.generate_embed_passmanager(cmap)
        first_layout_circ = initial_pm.run(qc)
        out_pass = ApplyLayout()
        out_pass.property_set["layout"] = first_layout_circ.layout.initial_layout
        out_pass.property_set[
            "original_qubit_indices"
        ] = first_layout_circ.layout.input_qubit_mapping
        out_pass.property_set["final_layout"] = Layout(
            {
                first_layout_circ.qubits[0]: 0,
                first_layout_circ.qubits[1]: 3,
                first_layout_circ.qubits[2]: 2,
                first_layout_circ.qubits[3]: 4,
                first_layout_circ.qubits[4]: 1,
            }
        )
        # Set a post layout like vf2postlayout would:
        out_pass.property_set["post_layout"] = Layout(
            {
                first_layout_circ.qubits[0]: 0,
                first_layout_circ.qubits[2]: 4,
                first_layout_circ.qubits[1]: 2,
                first_layout_circ.qubits[3]: 1,
                first_layout_circ.qubits[4]: 3,
            }
        )
        out_pass(first_layout_circ)
        self.assertEqual(
            out_pass.property_set["final_layout"],
            Layout(
                {
                    first_layout_circ.qubits[0]: 0,
                    first_layout_circ.qubits[2]: 1,
                    first_layout_circ.qubits[4]: 4,
                    first_layout_circ.qubits[1]: 3,
                    first_layout_circ.qubits[3]: 2,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
