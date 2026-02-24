from test import QiskitTestCase
import numpy as np
from qiskit import QuantumCircuit


class TestGateLabels(QiskitTestCase):
    def test_gate_labels_are_applied(self):
        qc = QuantumCircuit(2)
        # Make gates with labels
        qc.rz(np.pi, 0, label="rz_gate")
        qc.rxx(np.pi, 0, 1, "rxx_gate")
        qc.s(0, "s_gate")

        # Check labels are there
        assert qc.data[0].label == "rz_gate"
        assert qc.data[1].label == "rxx_gate"
        assert qc.data[2].label == "s_gate"
