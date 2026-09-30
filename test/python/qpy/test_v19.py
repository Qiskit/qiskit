# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPY v19 format changes and non-standard gate attributes."""

import io
import warnings

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library import StatePreparation, MCXVChain
from qiskit.qpy import dump, load
from qiskit.qpy.binary_io import circuits
from test import QiskitTestCase


def _dump_load(qc: QuantumCircuit, version: int | None = None) -> QuantumCircuit:
    buf = io.BytesIO()
    dump(qc, buf, version=version)
    buf.seek(0)
    return load(buf)[0]


class TestQPYVersion19(QiskitTestCase):
    """Test QPY v19 support for gates with non-standard fields."""

    def test_state_preparation_inverse_roundtrip_v19(self):
        """Test StatePreparation with inverse=True roundtrips and preserves _inverse in v19."""
        sp = StatePreparation([0, 1], inverse=True)
        self.assertTrue(sp._inverse)

        qc = QuantumCircuit(1)
        qc.append(sp, [0])

        loaded_qc = _dump_load(qc, version=19)
        loaded_sp = loaded_qc.data[0].operation
        self.assertIsInstance(loaded_sp, StatePreparation)
        self.assertTrue(loaded_sp._inverse)
        self.assertEqual(qc, loaded_qc)

    def test_state_preparation_inverse_difference_v18_vs_v19(self):
        """Demonstrate that v18 loses _inverse while v19 preserves it (#17000)."""
        sp = StatePreparation([0, 1], inverse=True)
        qc = QuantumCircuit(1)
        qc.append(sp, [0])

        # In QPY v18, _inverse was not preserved
        loaded_v18 = _dump_load(qc, version=18)
        self.assertFalse(loaded_v18.data[0].operation._inverse)

        # In QPY v19, _inverse is preserved
        loaded_v19 = _dump_load(qc, version=19)
        self.assertTrue(loaded_v19.data[0].operation._inverse)

    def test_state_preparation_non_inverse_roundtrip(self):
        """Test StatePreparation with inverse=False roundtrips properly."""
        sp = StatePreparation([1 / 2**0.5, 1 / 2**0.5], inverse=False)
        self.assertFalse(sp._inverse)

        qc = QuantumCircuit(1)
        qc.append(sp, [0])

        loaded_qc = _dump_load(qc, version=19)
        loaded_sp = loaded_qc.data[0].operation
        self.assertIsInstance(loaded_sp, StatePreparation)
        self.assertFalse(loaded_sp._inverse)
        self.assertEqual(qc, loaded_qc)

    def test_state_preparation_with_labels_and_integers(self):
        """Test StatePreparation initialized with string labels and ints."""
        qc = QuantumCircuit(2)
        sp1 = StatePreparation("10", inverse=True)
        sp2 = StatePreparation(3, 2, inverse=True)
        qc.append(sp1, [0, 1])
        qc.append(sp2, [0, 1])

        loaded_qc = _dump_load(qc, version=19)
        self.assertTrue(loaded_qc.data[0].operation._inverse)
        self.assertTrue(loaded_qc.data[1].operation._inverse)
        self.assertEqual(qc, loaded_qc)

    def test_mcx_vchain_attributes_roundtrip_v19(self):
        """Test MCXVChain preserves dirty_ancillas, relative_phase, action_only in QPY 19."""
        cases = [
            {"dirty_ancillas": True, "relative_phase": False, "action_only": False},
            {"dirty_ancillas": False, "relative_phase": True, "action_only": False},
            {"dirty_ancillas": False, "relative_phase": False, "action_only": True},
            {"dirty_ancillas": True, "relative_phase": True, "action_only": True},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            for kwargs in cases:
                with self.subTest(**kwargs):
                    mcx = MCXVChain(num_ctrl_qubits=3, **kwargs)
                    qc = QuantumCircuit(5)
                    qc.append(mcx, [0, 1, 2, 3, 4])

                    loaded_qc = _dump_load(qc, version=19)
                    loaded_mcx = loaded_qc.data[0].operation
                    self.assertIsInstance(loaded_mcx, MCXVChain)
                    self.assertEqual(loaded_mcx._dirty_ancillas, kwargs["dirty_ancillas"])
                    self.assertEqual(loaded_mcx._relative_phase, kwargs["relative_phase"])
                    self.assertEqual(loaded_mcx._action_only, kwargs["action_only"])
                    self.assertEqual(qc, loaded_qc)

    def test_mcx_vchain_difference_v18_vs_v19(self):
        """Demonstrate that v18 loses MCXVChain fields while v19 preserves them (#11377)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            mcx = MCXVChain(num_ctrl_qubits=3, dirty_ancillas=True, relative_phase=True, action_only=True)
            qc = QuantumCircuit(5)
            qc.append(mcx, [0, 1, 2, 3, 4])

            # In QPY v18, these extra attributes were lost
            loaded_v18 = _dump_load(qc, version=18)
            op_v18 = loaded_v18.data[0].operation
            self.assertFalse(op_v18._dirty_ancillas)
            self.assertFalse(op_v18._relative_phase)
            self.assertFalse(op_v18._action_only)

            # In QPY v19, all attributes are preserved
            loaded_v19 = _dump_load(qc, version=19)
            op_v19 = loaded_v19.data[0].operation
            self.assertTrue(op_v19._dirty_ancillas)
            self.assertTrue(op_v19._relative_phase)
            self.assertTrue(op_v19._action_only)

    def test_python_instruction_codec_direct(self):
        """Test the Python instruction-level codec (_write_instruction and _read_instruction) for extra_data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            sp = StatePreparation([0, 1], inverse=True)
            mcx = MCXVChain(num_ctrl_qubits=3, dirty_ancillas=True, action_only=True)

            for op in (sp, mcx):
                qc = QuantumCircuit(op.num_qubits)
                inst = CircuitInstruction(op, tuple(qc.qubits))
                index_map = {"q": {q: i for i, q in enumerate(qc.qubits)}, "c": {}}
                annotation_state = circuits._AnnotationSerializationState({})

                buf = io.BytesIO()
                circuits._write_instruction(
                    buf,
                    inst,
                    custom_operations={},
                    index_map=index_map,
                    use_symengine=False,
                    version=19,
                    standalone_var_indices={},
                    annotation_state=annotation_state,
                )

                buf.seek(0)
                read_qc = QuantumCircuit(op.num_qubits)
                circuits._read_instruction(
                    buf,
                    version=19,
                    vectors={},
                    circuit=read_qc,
                    custom_operations={},
                    registers={"q": {}, "c": {}},
                    use_symengine=False,
                    standalone_vars=[],
                    annotation_state=annotation_state,
                )

                read_op = read_qc.data[0].operation
                if isinstance(op, StatePreparation):
                    self.assertTrue(read_op._inverse)
                else:
                    self.assertTrue(read_op._dirty_ancillas)
                    self.assertTrue(read_op._action_only)

    def test_backward_compatibility_v18_and_v17(self):
        """Test loading circuits written with QPY v17 and v18."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        for v in (17, 18):
            with self.subTest(version=v):
                loaded_qc = _dump_load(qc, version=v)
                self.assertEqual(qc, loaded_qc)
