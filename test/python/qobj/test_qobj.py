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

import copy

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble

from qiskit.qobj import (
    QasmQobj,
    PulseQobj,
    QobjHeader,
    PulseQobjInstruction,
    PulseQobjExperiment,
    PulseQobjConfig,
    QobjMeasurementOption,
    PulseLibraryItem,
    QasmQobjInstruction,
    QasmQobjExperiment,
    QasmQobjConfig,
    QasmExperimentCalibrations,
    GateCalibration,
)

from qiskit.test import QiskitTestCase


class TestQASMQobj(QiskitTestCase):
    """Tests for QasmQobj."""

    def setUp(self):
        super().setUp()
        self.valid_qobj = QasmQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=QasmQobjConfig(shots=1024, memory_slots=2, max_credits=10),
            experiments=[
                QasmQobjExperiment(
                    instructions=[
                        QasmQobjInstruction(name="u1", qubits=[1], params=[0.4]),
                        QasmQobjInstruction(name="u2", qubits=[1], params=[0.4, 0.2]),
                    ]
                )
            ],
        )

        self.valid_dict = {
            "qobj_id": "12345",
            "type": "QASM",
            "schema_version": "1.2.0",
            "header": {},
            "config": {"max_credits": 10, "memory_slots": 2, "shots": 1024},
            "experiments": [
                {
                    "instructions": [
                        {"name": "u1", "params": [0.4], "qubits": [1]},
                        {"name": "u2", "params": [0.4, 0.2], "qubits": [1]},
                    ]
                }
            ],
        }

        self.bad_qobj = copy.deepcopy(self.valid_qobj)
        self.bad_qobj.experiments = []

    def test_from_dict_per_class(self):
        """Test Qobj and its subclass representations given a dictionary."""
        test_parameters = {
            QasmQobj: (self.valid_qobj, self.valid_dict),
            QasmQobjConfig: (
                QasmQobjConfig(shots=1, memory_slots=2),
                {"shots": 1, "memory_slots": 2},
            ),
            QasmQobjExperiment: (
                QasmQobjExperiment(
                    instructions=[QasmQobjInstruction(name="u1", qubits=[1], params=[0.4])]
                ),
                {"instructions": [{"name": "u1", "qubits": [1], "params": [0.4]}]},
            ),
            QasmQobjInstruction: (
                QasmQobjInstruction(name="u1", qubits=[1], params=[0.4]),
                {"name": "u1", "qubits": [1], "params": [0.4]},
            ),
        }

        for qobj_class, (qobj_item, expected_dict) in test_parameters.items():
            with self.subTest(msg=str(qobj_class)):
                self.assertEqual(qobj_item, qobj_class.from_dict(expected_dict))

    def test_snapshot_instruction_to_dict(self):
        """Test snapshot instruction to dict."""
        valid_qobj = QasmQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=QasmQobjConfig(shots=1024, memory_slots=2, max_credits=10),
            experiments=[
                QasmQobjExperiment(
                    instructions=[
                        QasmQobjInstruction(name="u1", qubits=[1], params=[0.4]),
                        QasmQobjInstruction(name="u2", qubits=[1], params=[0.4, 0.2]),
                        QasmQobjInstruction(
                            name="snapshot",
                            qubits=[1],
                            snapshot_type="statevector",
                            label="my_snap",
                        ),
                    ]
                )
            ],
        )
        res = valid_qobj.to_dict()
        expected_dict = {
            "qobj_id": "12345",
            "type": "QASM",
            "schema_version": "1.3.0",
            "header": {},
            "config": {"max_credits": 10, "memory_slots": 2, "shots": 1024},
            "experiments": [
                {
                    "instructions": [
                        {"name": "u1", "params": [0.4], "qubits": [1]},
                        {"name": "u2", "params": [0.4, 0.2], "qubits": [1]},
                        {
                            "name": "snapshot",
                            "qubits": [1],
                            "snapshot_type": "statevector",
                            "label": "my_snap",
                        },
                    ],
                    "config": {},
                    "header": {},
                }
            ],
        }
        self.assertEqual(expected_dict, res)

    def test_snapshot_instruction_from_dict(self):
        """Test snapshot instruction from dict."""
        expected_qobj = QasmQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=QasmQobjConfig(shots=1024, memory_slots=2, max_credits=10),
            experiments=[
                QasmQobjExperiment(
                    instructions=[
                        QasmQobjInstruction(name="u1", qubits=[1], params=[0.4]),
                        QasmQobjInstruction(name="u2", qubits=[1], params=[0.4, 0.2]),
                        QasmQobjInstruction(
                            name="snapshot",
                            qubits=[1],
                            snapshot_type="statevector",
                            label="my_snap",
                        ),
                    ]
                )
            ],
        )
        qobj_dict = {
            "qobj_id": "12345",
            "type": "QASM",
            "schema_version": "1.2.0",
            "header": {},
            "config": {"max_credits": 10, "memory_slots": 2, "shots": 1024},
            "experiments": [
                {
                    "instructions": [
                        {"name": "u1", "params": [0.4], "qubits": [1]},
                        {"name": "u2", "params": [0.4, 0.2], "qubits": [1]},
                        {
                            "name": "snapshot",
                            "qubits": [1],
                            "snapshot_type": "statevector",
                            "label": "my_snap",
                        },
                    ]
                }
            ],
        }
        self.assertEqual(expected_qobj, QasmQobj.from_dict(qobj_dict))

    def test_change_qobj_after_compile(self):
        """Test modifying Qobj parameters after compile."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[0], qr[2])
        qc2.h(qr)
        qc1.measure(qr, cr)
        qc2.measure(qr, cr)
        circuits = [qc1, qc2]
        qobj1 = assemble(circuits, shots=1024, seed=88)
        qobj1.experiments[0].config.shots = 50
        qobj1.experiments[1].config.shots = 1
        self.assertTrue(qobj1.experiments[0].config.shots == 50)
        self.assertTrue(qobj1.experiments[1].config.shots == 1)
        self.assertTrue(qobj1.config.shots == 1024)

    def test_gate_calibrations_to_dict(self):
        """Test gate calibrations to dict."""

        pulse_library = [PulseLibraryItem(name="test", samples=[1j, 1j])]
        valid_qobj = QasmQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=QasmQobjConfig(
                shots=1024, memory_slots=2, max_credits=10, pulse_library=pulse_library
            ),
            experiments=[
                QasmQobjExperiment(
                    instructions=[QasmQobjInstruction(name="u1", qubits=[1], params=[0.4])],
                    config=QasmQobjConfig(
                        calibrations=QasmExperimentCalibrations(
                            gates=[
                                GateCalibration(
                                    name="u1", qubits=[1], params=[0.4], instructions=[]
                                )
                            ]
                        )
                    ),
                )
            ],
        )
        res = valid_qobj.to_dict()
        expected_dict = {
            "qobj_id": "12345",
            "type": "QASM",
            "schema_version": "1.3.0",
            "header": {},
            "config": {
                "max_credits": 10,
                "memory_slots": 2,
                "shots": 1024,
                "pulse_library": [{"name": "test", "samples": [1j, 1j]}],
            },
            "experiments": [
                {
                    "instructions": [{"name": "u1", "params": [0.4], "qubits": [1]}],
                    "config": {
                        "calibrations": {
                            "gates": [
                                {"name": "u1", "qubits": [1], "params": [0.4], "instructions": []}
                            ]
                        }
                    },
                    "header": {},
                }
            ],
        }
        self.assertEqual(expected_dict, res)


class TestPulseQobj(QiskitTestCase):
    """Tests for PulseQobj."""

    def setUp(self):
        super().setUp()
        self.valid_qobj = PulseQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=PulseQobjConfig(
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
            ),
            experiments=[
                PulseQobjExperiment(
                    instructions=[
                        PulseQobjInstruction(name="pulse0", t0=0, ch="d0"),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase=1.57),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase=0.0),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase="P1"),
                        PulseQobjInstruction(name="setp", t0=10, ch="d0", phase=3.14),
                        PulseQobjInstruction(name="setf", t0=10, ch="d0", frequency=8.0),
                        PulseQobjInstruction(name="shiftf", t0=10, ch="d0", frequency=4.0),
                        PulseQobjInstruction(
                            name="acquire",
                            t0=15,
                            duration=5,
                            qubits=[0],
                            memory_slot=[0],
                            kernels=[
                                QobjMeasurementOption(
                                    name="boxcar", params={"start_window": 0, "stop_window": 5}
                                )
                            ],
                        ),
                    ]
                )
            ],
        )
        self.valid_dict = {
            "qobj_id": "12345",
            "type": "PULSE",
            "schema_version": "1.2.0",
            "header": {},
            "config": {
                "max_credits": 10,
                "memory_slots": 2,
                "shots": 1024,
                "meas_level": 1,
                "memory_slot_size": 8192,
                "meas_return": "avg",
                "pulse_library": [{"name": "pulse0", "samples": [0, 0.5, 0]}],
                "qubit_lo_freq": [4.9],
                "meas_lo_freq": [6.9],
                "rep_time": 1000,
            },
            "experiments": [
                {
                    "instructions": [
                        {"name": "pulse0", "t0": 0, "ch": "d0"},
                        {"name": "fc", "t0": 5, "ch": "d0", "phase": 1.57},
                        {"name": "fc", "t0": 5, "ch": "d0", "phase": 0},
                        {"name": "fc", "t0": 5, "ch": "d0", "phase": "P1"},
                        {"name": "setp", "t0": 10, "ch": "d0", "phase": 3.14},
                        {"name": "setf", "t0": 10, "ch": "d0", "frequency": 8.0},
                        {"name": "shiftf", "t0": 10, "ch": "d0", "frequency": 4.0},
                        {
                            "name": "acquire",
                            "t0": 15,
                            "duration": 5,
                            "qubits": [0],
                            "memory_slot": [0],
                            "kernels": [
                                {"name": "boxcar", "params": {"start_window": 0, "stop_window": 5}}
                            ],
                        },
                    ]
                }
            ],
        }

    def test_from_dict_per_class(self):
        """Test converting to Qobj and its subclass representations given a dictionary."""
        test_parameters = {
            PulseQobj: (self.valid_qobj, self.valid_dict),
            PulseQobjConfig: (
                PulseQobjConfig(
                    meas_level=1,
                    memory_slot_size=8192,
                    meas_return="avg",
                    pulse_library=[PulseLibraryItem(name="pulse0", samples=[0.1 + 0.0j])],
                    qubit_lo_freq=[4.9],
                    meas_lo_freq=[6.9],
                    rep_time=1000,
                ),
                {
                    "meas_level": 1,
                    "memory_slot_size": 8192,
                    "meas_return": "avg",
                    "pulse_library": [{"name": "pulse0", "samples": [0.1 + 0j]}],
                    "qubit_lo_freq": [4.9],
                    "meas_lo_freq": [6.9],
                    "rep_time": 1000,
                },
            ),
            PulseLibraryItem: (
                PulseLibraryItem(name="pulse0", samples=[0.1 + 0.0j]),
                {"name": "pulse0", "samples": [0.1 + 0j]},
            ),
            PulseQobjExperiment: (
                PulseQobjExperiment(
                    instructions=[PulseQobjInstruction(name="pulse0", t0=0, ch="d0")]
                ),
                {"instructions": [{"name": "pulse0", "t0": 0, "ch": "d0"}]},
            ),
            PulseQobjInstruction: (
                PulseQobjInstruction(name="pulse0", t0=0, ch="d0"),
                {"name": "pulse0", "t0": 0, "ch": "d0"},
            ),
        }

        for qobj_class, (qobj_item, expected_dict) in test_parameters.items():
            with self.subTest(msg=str(qobj_class)):
                self.assertEqual(qobj_item, qobj_class.from_dict(expected_dict))

    def test_to_dict_per_class(self):
        """Test converting from Qobj and its subclass representations given a dictionary."""
        test_parameters = {
            PulseQobj: (self.valid_qobj, self.valid_dict),
            PulseQobjConfig: (
                PulseQobjConfig(
                    meas_level=1,
                    memory_slot_size=8192,
                    meas_return="avg",
                    pulse_library=[PulseLibraryItem(name="pulse0", samples=[0.1 + 0.0j])],
                    qubit_lo_freq=[4.9],
                    meas_lo_freq=[6.9],
                    rep_time=1000,
                ),
                {
                    "meas_level": 1,
                    "memory_slot_size": 8192,
                    "meas_return": "avg",
                    "pulse_library": [{"name": "pulse0", "samples": [0.1 + 0j]}],
                    "qubit_lo_freq": [4.9],
                    "meas_lo_freq": [6.9],
                    "rep_time": 1000,
                },
            ),
            PulseLibraryItem: (
                PulseLibraryItem(name="pulse0", samples=[0.1 + 0.0j]),
                {"name": "pulse0", "samples": [0.1 + 0j]},
            ),
            PulseQobjExperiment: (
                PulseQobjExperiment(
                    instructions=[PulseQobjInstruction(name="pulse0", t0=0, ch="d0")]
                ),
                {"instructions": [{"name": "pulse0", "t0": 0, "ch": "d0"}]},
            ),
            PulseQobjInstruction: (
                PulseQobjInstruction(name="pulse0", t0=0, ch="d0"),
                {"name": "pulse0", "t0": 0, "ch": "d0"},
            ),
        }

        for qobj_class, (qobj_item, expected_dict) in test_parameters.items():
            with self.subTest(msg=str(qobj_class)):
                self.assertEqual(qobj_item.to_dict(), expected_dict)


def _nop():
    pass
