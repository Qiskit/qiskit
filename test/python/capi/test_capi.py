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

import ctypes
from pathlib import Path

from qiskit import capi
from qiskit.circuit import Gate, Instruction
from qiskit.transpiler import Target
from test import QiskitTestCase


class TestCAPI(QiskitTestCase):
    def test_includes_exists(self):
        path = Path(capi.get_include())
        self.assertIn("QISKIT_H", (path / "qiskit.h").read_text(encoding="utf-8"))
        self.assertLess(set(), set((path / "qiskit").glob("*.h")))

    def test_library_exists(self):
        self.assertTrue(Path(capi.get_lib()).exists())

    def test_python_target_operations(self):
        for operation in [Gate("custom", 1, []), Instruction("custom", 1, 0, [0.25, object()])]:
            with self.subTest(operation=type(operation).__name__):
                target = Target(num_qubits=1)
                target.add_instruction(operation, {(0,): None})
                native = capi.qk_target_borrow_from_python(target)
                result = capi.QkTargetOp()
                capi.qk_target_op_get(native, 0, ctypes.byref(result))
                self.addCleanup(capi.qk_target_op_clear, ctypes.byref(result))
                self.assertEqual(result.op_type, capi.QkOperationKind.Unknown.value.value)
                self.assertEqual(ctypes.string_at(result.name), b"custom")
                self.assertEqual(result.num_qubits, 1)
                self.assertEqual(result.num_params, len(operation.params))
                if result.num_params:
                    self.assertEqual(capi.qk_param_as_real(result.params[0]), 0.25)
                    self.assertEqual(
                        capi.qk_param_kind(result.params[1]), capi.QkParamKind.Unknown.value.value
                    )
