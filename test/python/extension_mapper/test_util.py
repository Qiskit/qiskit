"""Test cases for the util package"""
from unittest import TestCase

from qiskit.dagcircuit import DAGCircuit

from src import util


class TestUtil(TestCase):
    """The test cases."""
    def setUp(self) -> None:
        self.circuit = TestUtil.basic_dag()

    def test_empty_circuit_simple(self) -> None:
        """Test whether empty circuit does produce an empty circuit from a simple DAG."""
        self.circuit.add_creg('c', 2)
        self.circuit.add_qreg('q', 2)
        self.circuit.apply_operation_back("cx", [("q", 0), ("q", 1)], condition=("c", 0))
        emptied_circuit = util.empty_circuit(self.circuit)

        # The parameters of the emptied circuit should be the same
        self.assertEqual(self.circuit.input_map, emptied_circuit.input_map)
        self.assertEqual(self.circuit.output_map, emptied_circuit.output_map)
        self.assertEqual(self.circuit.basis, emptied_circuit.basis)
        self.assertEqual(self.circuit.gates, emptied_circuit.gates)
        op_nodes = list(
            filter(lambda n: n[1]["type"] == "op", emptied_circuit.multi_graph.nodes(data=True)))
        self.assertEqual(0, len(op_nodes))

    def test_empty_circuit_qregs(self) -> None:
        self.circuit.add_creg('c', 2)
        self.circuit.add_qreg('q', 2)
        self.circuit.apply_operation_back("cx", [("q", 0), ("q", 1)], condition=("c", 0))
        emptied_circuit = util.empty_circuit(self.circuit, qregs=[("0", 1), ("1", 1)])

        # The emptied circuit should only have the specified qregs.
        self.assertEqual(2, len(emptied_circuit.qregs))
        qregs = set(emptied_circuit.qregs.items())
        self.assertEqual({("0", 1), ("1", 1)}, qregs)

    @staticmethod
    def basic_dag() -> DAGCircuit:
        """Create a DAGCircuit that supports cx, u2 and measure operations."""
        circuit = DAGCircuit()
        circuit.add_basis_element('cx', 2)
        circuit.add_basis_element('u2', 1, number_parameters=2)
        circuit.add_basis_element('measure', 1, number_classical=1)
        circuit.add_basis_element('barrier', -1)  # nr of qargs is ignored for "barrier"
        return circuit
