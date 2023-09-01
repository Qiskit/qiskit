# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object

from qiskit.compiler import assemble
from qiskit.assembler import disassemble

from .utils import random_circuit


class AssemblerBenchmarks:
    params = ([8], [4096], [1, 100])
    param_names = ["n_qubits", "depth", "number of circuits"]
    timeout = 600
    version = 2

    def setup(self, n_qubits, depth, number_of_circuits):
        seed = 42
        self.circuit = random_circuit(n_qubits, depth, measure=True, conditional=True, seed=seed)
        self.circuits = [self.circuit] * number_of_circuits

    def time_assemble_circuit(self, _, __, ___):
        assemble(self.circuits)


class DisassemblerBenchmarks:
    params = ([8], [4096], [1, 100])
    param_names = ["n_qubits", "depth", "number of circuits"]
    timeout = 600

    def setup(self, n_qubits, depth, number_of_circuits):
        seed = 424242
        self.circuit = random_circuit(n_qubits, depth, measure=True, conditional=True, seed=seed)
        self.circuits = [self.circuit] * number_of_circuits
        self.qobj = assemble(self.circuits)

    def time_disassemble_circuit(self, _, __, ___):
        disassemble(self.qobj)
