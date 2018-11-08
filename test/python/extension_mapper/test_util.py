"""Test cases for the util package"""
from unittest import TestCase

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.passes.extension_mapper.src import util


class TestUtil(TestCase):
    """The test cases."""
    def setUp(self):
        self.circuit = TestUtil.basic_dag()

    def test_simple_empty_circuit(self):
        """Test whether empty circuit does produce an empty circuit from a simple DAG."""
        self.circuit.add_creg(ClassicalRegister(2, name="c"))
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
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

    def test_empty_circuit_qregs(self):
        """Test whether empty circuit preserves provided qregs."""
        self.circuit.add_creg(ClassicalRegister(2, name="c"))
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back("cx", [("q", 0), ("q", 1)], condition=("c", 0))
        qregs_in = {"q0": QuantumRegister(1, name="q0"), "q1": QuantumRegister(1, name="q1")}
        emptied_circuit = util.empty_circuit(self.circuit, qregs=qregs_in)

        # The emptied circuit should only have the specified qregs.
        self.assertEqual(2, len(emptied_circuit.qregs))
        qregs_out = set(emptied_circuit.qregs.items())
        self.assertEqual({(name, qreg) for name, qreg in qregs_in.items()}, qregs_out)

    @staticmethod
    def basic_dag() -> DAGCircuit:
        """Create a DAGCircuit that supports cx, u2 and measure operations."""
        circuit = DAGCircuit()
        circuit.add_basis_element('cx', 2)
        circuit.add_basis_element('u2', 1, number_parameters=2)
        circuit.add_basis_element('measure', 1, number_classical=1)
        circuit.add_basis_element('barrier', -1)  # nr of qargs is ignored for "barrier"
        return circuit
