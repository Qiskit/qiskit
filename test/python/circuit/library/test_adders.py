# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test adder circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
)
from qiskit.synthesis.arithmetic import adder_ripple_c04, adder_ripple_v95, adder_qft_d00
from qiskit.transpiler.passes import HLSConfig, HighLevelSynthesis
from test import QiskitTestCase  # pylint: disable=wrong-import-order

ADDERS = {
    "vbe": adder_ripple_v95,
    "cdkm": adder_ripple_c04,
    "draper": adder_qft_d00,
}

ADDER_CIRCUITS = {
    "vbe": VBERippleCarryAdder,
    "cdkm": CDKMRippleCarryAdder,
    "draper": DraperQFTAdder,
}


@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(
        self, num_state_qubits: int, adder: QuantumCircuit, inplace: bool, kind: str
    ):
        """Assert that adder correctly implements the summation.

        This test prepares a equal superposition state in both input registers, then performs
        the addition on the superposition and checks that the output state is the expected
        superposition of all possible additions.

        Args:
            num_state_qubits: The number of bits in the numbers that are added.
            adder: The circuit performing the addition of two numbers with ``num_state_qubits``
                bits.
            inplace: If True, compare against an inplace addition where the result is written into
                the second register plus carry qubit. If False, assume that the result is written
                into a third register of appropriate size.
            kind: The kind of adder; "fixed", "half", or "full".
        """
        circuit = QuantumCircuit(*adder.qregs)

        # create equal superposition
        if kind == "full":
            num_superpos_qubits = 2 * num_state_qubits + 1
        else:
            num_superpos_qubits = 2 * num_state_qubits
        circuit.h(range(num_superpos_qubits))

        # apply adder circuit
        circuit.compose(adder, inplace=True)

        # obtain the statevector and the probabilities, we don't trace out the ancilla qubits
        # as we verify that all ancilla qubits have been uncomputed to state 0 again
        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        pad = "0" * circuit.num_ancillas  # state of the ancillas

        # compute the expected results
        expectations = np.zeros_like(probabilities)
        num_bits_sum = num_state_qubits + 1
        # iterate over all possible inputs
        for x in range(2**num_state_qubits):
            for y in range(2**num_state_qubits):
                # compute the sum
                if kind == "full":
                    additions = [x + y, 1 + x + y]
                elif kind == "half":
                    additions = [x + y]
                else:
                    additions = [(x + y) % (2**num_state_qubits)]

                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)

                for i, addition in enumerate(additions):
                    bin_res = bin(addition)[2:].zfill(num_bits_sum)
                    if kind == "full":
                        cin = str(i)
                        bin_index = (
                            pad + bin_res + bin_x + cin
                            if inplace
                            else pad + bin_res + bin_y + bin_x + cin
                        )
                    else:
                        bin_index = (
                            pad + bin_res + bin_x if inplace else pad + bin_res + bin_y + bin_x
                        )

                    index = int(bin_index, 2)
                    expectations[index] += 1 / 2**num_superpos_qubits

        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (3, "cdkm", "half"),
        (5, "cdkm", "half"),
        (3, "cdkm", "fixed"),
        (5, "cdkm", "fixed"),
        (1, "cdkm", "full"),
        (3, "cdkm", "full"),
        (5, "cdkm", "full"),
        (3, "draper", "half"),
        (5, "draper", "half"),
        (3, "draper", "fixed"),
        (5, "draper", "fixed"),
        (1, "vbe", "full"),
        (3, "vbe", "full"),
        (5, "vbe", "full"),
        (1, "vbe", "half"),
        (2, "vbe", "half"),
        (5, "vbe", "half"),
        (1, "vbe", "fixed"),
        (2, "vbe", "fixed"),
        (4, "vbe", "fixed"),
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, kind):
        """Test summation for all implemented adders."""
        for use_function in [True, False]:
            with self.subTest(use_function=use_function):
                if use_function:
                    circuit = ADDERS[adder](num_state_qubits, kind)
                else:
                    circuit = ADDER_CIRCUITS[adder](num_state_qubits, kind)

        self.assertAdditionIsCorrect(num_state_qubits, circuit, True, kind)

    @data(
        CDKMRippleCarryAdder,
        DraperQFTAdder,
        VBERippleCarryAdder,
        adder_ripple_c04,
        adder_ripple_v95,
        adder_qft_d00,
    )
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)

    def test_plugins(self):
        """Test setting the HLS plugins for the modular adder."""

        # all gates with the plugins we check
        modes = {
            "ModularAdder": (ModularAdderGate, ["ripple_c04", "ripple_v95", "qft_d00"]),
            "HalfAdder": (HalfAdderGate, ["ripple_c04", "ripple_v95", "qft_d00"]),
            "FullAdder": (FullAdderGate, ["ripple_c04", "ripple_v95"]),
        }

        # an operation we expect to be in the circuit with given plugin name
        expected_ops = {
            "ripple_c04": "MAJ",
            "ripple_v95": "Carry",
            "qft_d00": "cp",
        }

        num_state_qubits = 3
        max_auxiliaries = num_state_qubits - 1  # V95 needs these
        max_num_qubits = 2 * num_state_qubits + max_auxiliaries + 2

        for name, (adder_cls, plugins) in modes.items():
            for plugin in plugins:
                with self.subTest(name=name, plugin=plugin):
                    adder = adder_cls(num_state_qubits)

                    circuit = QuantumCircuit(max_num_qubits)
                    circuit.append(adder, range(adder.num_qubits))

                    hls_config = HLSConfig(**{name: [plugin]})
                    hls = HighLevelSynthesis(hls_config=hls_config)

                    synth = hls(circuit)
                    ops = set(synth.count_ops().keys())

                    self.assertTrue(expected_ops[plugin] in ops)


if __name__ == "__main__":
    unittest.main()
