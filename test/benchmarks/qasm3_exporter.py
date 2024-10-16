# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring
# pylint: disable=attribute-defined-outside-init

from qiskit import qasm3

from .utils import random_circuit


class QASM3ExporterBenchmarks:

    params = ([20], [256, 1024], [0, 42])

    param_names = ["n_qubits", "depth", "seed"]
    timeout = 300

    def setup(self, n_qubits, depth, seed):
        self.circuit = random_circuit(
            n_qubits,
            depth,
            measure=True,
            conditional=True,
            reset=True,
            seed=seed,
            max_operands=3,
        )

    def time_dumps(self, _, __):
        ___ = qasm3.dumps(self.circuit)
