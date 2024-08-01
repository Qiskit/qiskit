# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing a Faulty 7QV1Pulse Backend."""

from qiskit.providers.backend_compat import convert_to_target
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from .faulty_backends import (
    Fake7QV1FaultyCX01CX10,
    Fake7QV1FaultyQ1,
    Fake7QV1MissingQ1Property,
    Fake7QV1FaultyCX13CX31,
)


class FaultyQubitBackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with Fake7QV1FaultyQ1,
    which is like Fake7QV1 but with a faulty 1Q"""

    # These test can be removed with Fake7QV1FaultyQ1

    backend = Fake7QV1FaultyQ1()

    def test_operational_false(self):
        """Test operation status of the qubit. Q1 is non-operational"""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        self.assertFalse(properties.is_qubit_operational(1))

    def test_faulty_qubits(self):
        """Test faulty_qubits method."""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        self.assertEqual(properties.faulty_qubits(), [1])

    def test_convert_to_target_with_filter(self):
        """Test converting legacy data structure to V2 target model with faulty qubits.

        Measure and Delay are automatically added to the output Target
        even though instruction is not provided by the backend,
        since these are the necessary instructions that the transpiler may assume.
        """
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()

        # Filter out faulty Q1
        target = convert_to_target(
            configuration=self.backend.configuration(),
            properties=properties,
            add_delay=True,
            filter_faulty=True,
        )
        self.assertFalse(target.instruction_supported(operation_name="measure", qargs=(1,)))
        self.assertFalse(target.instruction_supported(operation_name="delay", qargs=(1,)))

    def test_convert_to_target_without_filter(self):
        """Test converting legacy data structure to V2 target model with faulty qubits."""

        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()

        # Include faulty Q1 even though data could be incomplete
        target = convert_to_target(
            configuration=self.backend.configuration(),
            properties=properties,
            add_delay=True,
            filter_faulty=False,
        )
        self.assertTrue(target.instruction_supported(operation_name="measure", qargs=(1,)))
        self.assertTrue(target.instruction_supported(operation_name="delay", qargs=(1,)))

        # Properties are preserved
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()

        self.assertEqual(
            target.qubit_properties[1].t1,
            properties.t1(1),
        )
        self.assertEqual(
            target.qubit_properties[1].t2,
            properties.t2(1),
        )
        self.assertEqual(
            target.qubit_properties[1].frequency,
            properties.frequency(1),
        )


class FaultyGate13BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with Fake7QV1FaultyCX13CX31,
    which is like Fake7QV1 but with a faulty CX(Q1, Q3) and symmetric."""

    backend = Fake7QV1FaultyCX13CX31()

    def test_operational_gate(self):
        """Test is_gate_operational method."""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        self.assertFalse(properties.is_gate_operational("cx", [1, 3]))
        self.assertFalse(properties.is_gate_operational("cx", [3, 1]))

    def test_faulty_gates(self):
        """Test faulty_gates method."""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        gates = properties.faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ["cx", "cx"])
        self.assertEqual(sorted(gate.qubits for gate in gates), [[1, 3], [3, 1]])


class FaultyGate01BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with Fake7QV1FaultyCX13CX31,
    which is like Fake7QV1 but with a faulty CX(Q1, Q3) and symmetric."""

    backend = Fake7QV1FaultyCX01CX10()

    def test_operational_gate(self):
        """Test is_gate_operational method."""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        self.assertFalse(properties.is_gate_operational("cx", [0, 1]))
        self.assertFalse(properties.is_gate_operational("cx", [1, 0]))

    def test_faulty_gates(self):
        """Test faulty_gates method."""
        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()
        gates = properties.faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ["cx", "cx"])
        self.assertEqual(sorted(gate.qubits for gate in gates), [[0, 1], [1, 0]])


class MissingPropertyQubitBackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with Fake7QV1MissingQ1Property,
    which is like Fake7QV1 but with Q1 with missing T1 property."""

    backend = Fake7QV1MissingQ1Property()

    def test_convert_to_target(self):
        """Test converting legacy data structure to V2 target model with missing qubit property."""

        with self.assertWarns(DeprecationWarning):
            properties = self.backend.properties()

        target = convert_to_target(
            configuration=self.backend.configuration(),
            properties=properties,
            add_delay=True,
            filter_faulty=True,
        )

        self.assertIsNone(target.qubit_properties[1].t1)
        self.assertEqual(
            target.qubit_properties[1].t2,
            properties.t2(1),
        )
        self.assertEqual(
            target.qubit_properties[1].frequency,
            properties.frequency(1),
        )
