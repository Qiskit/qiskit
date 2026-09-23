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

from pathlib import Path

from qiskit import capi
from qiskit.circuit import QuantumCircuit, Parameter
from test import QiskitTestCase
import ctypes


class TestCircuit(QiskitTestCase):
    """Unit tests for the circuit via the C FFI"""

    def test_circuit_with_multiple_dt_units(self):
        circ = QuantumCircuit(2, 0)
        circ.delay(100, 0, "dt")
        circ.h(0)
        circ.delay(734, 0, "ms")
        circ.cx(0, 1)
        circ.delay(Parameter("a"), 1, "ns")

        c_circ = capi.qk_circuit_borrow_from_python(circ._data)

        expected_param_types = [
            capi.QkParamKind.Int.value.value,
            capi.QkParamKind.Real.value.value,
            capi.QkParamKind.ParameterExpression.value.value,
        ]

        param_result_types = [100, 734.0]

        # Test each delay
        for idx, c_idx in enumerate([0, 2, 4]):
            view = capi.QkCircuitInstruction.from_buffer(bytearray(48))
            capi.qk_circuit_get_instruction(c_circ, c_idx, view)

            # Compare the element's kind using its integer value.
            param_kind = capi.qk_param_kind(view.params[0])
            self.assertEqual(capi.qk_param_kind(view.params[0]), expected_param_types[idx])

            # Check if the integer case happens and load the integer within a pointer.
            if param_kind == capi.QkParamKind.Int.value.value:
                val = ctypes.c_long.from_buffer(bytearray(64))
                capi.qk_param_as_int(view.params[0], val)

                self.assertEqual(val.value, param_result_types[idx])

            # Process differs with real as we don't really allocate.
            elif param_kind == capi.QkParamKind.Real.value.value:
                val = capi.qk_param_as_real(view.params[0])

                self.assertEqual(val, param_result_types[idx])
            # We don't yet have ways of obtaining expressions so we continue the loop by freeing.

            capi.qk_circuit_instruction_clear(view)
