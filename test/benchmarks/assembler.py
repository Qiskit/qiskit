# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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

from .utils import random_circuit


class AssemblerBenchmarks:
    params = ([1, 2, 5, 8],
              [8, 128, 1024, 2048, 4096])
    param_names = ['n_qubits', 'depth']
    timeout = 600

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(n_qubits, depth, measure=True,
                                      conditional=True, seed=seed)

    def time_assemble_circuit(self, _, __):
        assemble(self.circuit)
