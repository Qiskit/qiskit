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


"""Qobj tests."""


from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.compiler import assemble
from qiskit.qobj import PulseQobjConfig, PulseLibraryItem, QasmQobjConfig
from qiskit.test import QiskitTestCase
from qiskit.utils import QuantumInstance


class TestMaxCreditsDeprecated(QiskitTestCase):
    """Tests for max_credits deprecation."""

    def test_assemble_warns(self):
        """Test that assemble function displays a deprecation warning if max_credits is used"""
        circuit = QuantumCircuit(1, 1)
        with self.assertWarns(DeprecationWarning):
            assemble(circuit, max_credits=10)

    def test_execute_warns(self):
        """Test that execute function displays a deprecation warning if max_credits is used"""
        backend = BasicAer.get_backend("statevector_simulator")
        circuit = QuantumCircuit(1, 1)
        with self.assertWarns(DeprecationWarning):
            execute(circuit, backend, max_credits=10)

    def test_qasm_obj_config_warns(self):
        """Test that QasmQobjConfig constructor displays a deprecation
        warning if max_credits is used"""
        with self.assertWarns(DeprecationWarning):
            QasmQobjConfig(shots=1024, memory_slots=2, max_credits=10)

    def test_quantum_instance_warns(self):
        """Test that QuantumInstance constructor displays a deprecation
        warning if max_credits is used"""
        with self.assertWarns(DeprecationWarning):
            QuantumInstance(BasicAer.get_backend("statevector_simulator"), max_credits=10)

    def test_pulse_obj_config_warns(self):
        """Test that PulseQobjConfig constructor displays a deprecation
        warning if max_credits is used"""
        with self.assertWarns(DeprecationWarning):
            PulseQobjConfig(
                shots=1024,
                memory_slots=2,
                max_credits=10,
                meas_level=1,
                memory_slot_size=8192,
                meas_return="avg",
                pulse_library=[
                    PulseLibraryItem(name="pulse0", samples=[0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j])
                ],
                qubit_lo_freq=[4.9],
                meas_lo_freq=[6.9],
                rep_time=1000,
            )
