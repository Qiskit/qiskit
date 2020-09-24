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

"""Testing noise aware swap mapping"""

import unittest

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeManhattan
from qiskit.test.mock.fake_backend import FakeBackend


class BadManhattan(FakeBackend):
    """A fake Manhattan backend."""

    def __init__(self, config, props):
        configuration = config
        configuration.backend_name = 'bad_manhattan'
        self._defaults = None
        self._properties = props
        super().__init__(configuration)

    def properties(self):
        return self._properties


class TestNoiseSwap(QiskitTestCase):
    """ Tests for noise aware swap mapping"""

    def test_routing_bad_manhattan(self):
        """ Swap around Manhattan with bad cx edges"""

        man = FakeManhattan()

        bad_gates = [[kk, kk+1] for kk in range(13, 21)]
        bad_gates += [[kk, kk+1] for kk in range(44, 51)]
        rev_bad_gates = [kk[::-1] for kk in bad_gates]
        bad_gates += rev_bad_gates

        props = man.properties()

        for gate in props.gates:
            if gate.qubits in bad_gates:
                gate.parameters[0].value = 1.0

        bad = BadManhattan(man.configuration(), props)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        swap_methods = ['stochastic', 'sabre', 'lookahead']

        for method in swap_methods:

            new_qc = transpile(qc, bad,
                               initial_layout=[0, 64],
                               routing_method=method,
                               seed_transpiler=12345)

            any_bad = False
            for gate in new_qc.data:
                if gate[0].num_qubits == 2:
                    if [gate[1][0].index, gate[1][1].index] in bad_gates:
                        any_bad = True
                        break

            self.assertFalse(any_bad)


if __name__ == '__main__':
    unittest.main()
