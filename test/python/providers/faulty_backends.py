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

"""Faulty fake backends for testing"""

from qiskit.providers.models import BackendProperties
from qiskit.test.mock import FakeOurense


class FakeOurenseFaultyQ1(FakeOurense):
    """A fake 5 qubit backend, with a faulty q1
    0 ↔ (1) ↔ 3 ↔ 4
         ↕
         2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the qubit 1 as non-operational.
        """
        props = super().properties().to_dict()
        props["qubits"][1].append(
            {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
        )
        return BackendProperties.from_dict(props)


class FakeOurenseFaultyCX01CX10(FakeOurense):
    """A fake 5 qubit backend, with faulty CX(Q1, Q0) and CX(Q0, Q1)
    0 (↔) 1 ↔ 3 ↔ 4
          ↕
          2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q0, Q1) (and symmetric) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and set(gate["qubits"]) == {0, 1}:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)


class FakeOurenseFaultyCX13CX31(FakeOurense):
    """A fake 5 qubit backend, with faulty CX(Q1, Q3) and CX(Q3, Q1)
    0 ↔ 1 (↔) 3 ↔ 4
        ↕
        2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q1, Q3) (and symmetric) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and set(gate["qubits"]) == {3, 1}:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)


class FakeOurenseFaultyCX13(FakeOurense):
    """A fake 5 qubit backend, with faulty CX(Q1, Q3), but valid CX(Q3, Q1)
    0 ↔ 1 <- 3 ↔ 4
        ↕
        2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q1, Q3) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and gate["qubits"] == [1, 3]:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)
