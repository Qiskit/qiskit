import unittest
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info.states import Statevector
from qiskit.synthesis.multi_controlled import synth_mcx_1_clean_kg24, synth_mcx_2_clean_kg24, synth_mcx_1_dirty_kg24, synth_mcx_2_dirty_kg24
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestMCXSynth(QiskitTestCase):
    """Test the multi-controlled x circuit synthesis functions."""

    @data(3, 5, 8, 10, 15)
    def test_synth_mcx_1_ancilla(self, num_ctrl_qubits):
        """Test mcx synthesis with 1 ancilla."""
        reference = np.zeros(2 ** (num_ctrl_qubits + 1))
        reference[-1] = 1

        for gate_circuit in [
            synth_mcx_1_clean_kg24(num_ctrl_qubits),
            synth_mcx_1_dirty_kg24(num_ctrl_qubits),
        ]:
            with self.subTest(gate_circuit=gate_circuit):
                qc = QuantumCircuit(num_ctrl_qubits + 2)
                qc.x(list(range(num_ctrl_qubits)))
                qc.compose(gate_circuit, inplace=True)
                statevector = Statevector(qc).data

                corrected = np.zeros(2 ** (num_ctrl_qubits + 1), dtype=complex)
                for i, statevector_amplitude in enumerate(statevector):
                    i = int(bin(i)[2:].zfill(qc.num_qubits)[1 :], 2)    # 1 ancilla qubit
                    corrected[i] += statevector_amplitude
                statevector = corrected
            np.testing.assert_array_almost_equal(statevector.real, reference)
    
    @data(3, 5, 8, 10, 15)
    def test_synth_mcx_2_ancilla(self, num_ctrl_qubits):
        """Test mcx synthesis with 2 ancilla."""
        reference = np.zeros(2 ** (num_ctrl_qubits + 1))
        reference[-1] = 1

        for gate_circuit in [
            synth_mcx_2_clean_kg24(num_ctrl_qubits),
            synth_mcx_2_dirty_kg24(num_ctrl_qubits),
        ]:
            with self.subTest(gate_circuit=gate_circuit):
                qc = QuantumCircuit(num_ctrl_qubits + 3)
                qc.x(list(range(num_ctrl_qubits)))
                qc.compose(gate_circuit, inplace=True)
                statevector = Statevector(qc).data

                corrected = np.zeros(2 ** (num_ctrl_qubits + 1), dtype=complex)
                for i, statevector_amplitude in enumerate(statevector):
                    i = int(bin(i)[2:].zfill(qc.num_qubits)[2 :], 2)    # 2 ancilla qubits
                    corrected[i] += statevector_amplitude
                statevector = corrected
            np.testing.assert_array_almost_equal(statevector.real, reference)

    
if __name__ == "__main__":
    unittest.main()

