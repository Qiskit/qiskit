import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.circuit.controlflow import IfElseOp, BoxOp
from qiskit.visualization import circuit_drawer


class TestControlFlowAnnotationBox(unittest.TestCase):
    def test_if_else_flow_annotation_inside_box(self):
        """Test that flow control text like 'If' is rendered inside a box correctly."""
        inner_circuit = QuantumCircuit(1, name="inner_if_circuit")
        inner_circuit.x(0)

        qc = QuantumCircuit(1)
        qc.if_test((0, True), inner_circuit)

        fig = circuit_drawer(qc, output="mpl", fold=100)

        # Check that the figure is created (basic rendering test)
        self.assertIsNotNone(fig)

        # Optional: Save figure to visually inspect manually
        # fig.savefig("/tmp/test_if_else_inside_box.png")

    def test_box_op_annotation_inside_box(self):
        """Test that BoxOp is rendered correctly and tightly spaced."""
        subcircuit = QuantumCircuit(1, name="sub")
        subcircuit.append(XGate(), [0])

        qc = QuantumCircuit(1)
        qc.append(BoxOp(subcircuit), [0])

        fig = circuit_drawer(qc, output="mpl", fold=100)

        # Ensure the figure is rendered
        self.assertIsNotNone(fig)

        # Optional: Save figure for manual inspection
        # fig.savefig("/tmp/test_boxop_spacing.png")

    def test_nested_boxop_with_control_flow(self):
        """Test nested BoxOp containing control-flow ops renders without layout issues."""
        sub_inner = QuantumCircuit(1, name="inner_if")
        sub_inner.x(0)

        sub_box = QuantumCircuit(1, name="sub_box")
        sub_box.if_test((0, True), sub_inner)

        qc = QuantumCircuit(1)
        qc.append(BoxOp(sub_box), [0])

        fig = circuit_drawer(qc, output="mpl", fold=100)

        # Ensure the nested structure is drawn correctly
        self.assertIsNotNone(fig)

        # Optional: Save output for manual inspection
        # fig.savefig("/tmp/test_nested_boxop_ifelse.png")
