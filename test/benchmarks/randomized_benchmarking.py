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
# pylint: disable=import-error

import os
import numpy as np
from qiskit_experiments.library import StandardRB

try:
    from qiskit.compiler import transpile

    TRANSPILER_SEED_KEYWORD = "seed_transpiler"
except ImportError:
    from qiskit.transpiler import transpile

    TRANSPILER_SEED_KEYWORD = "seed_mapper"


def build_rb_circuit(qubits, length_vector, num_samples=1, seed=None):
    """
    Randomized Benchmarking sequences.
    """
    if not seed:
        np.random.seed(10)
    else:
        np.random.seed(seed)

    # Generate the sequences
    try:
        rb_exp = StandardRB(
            qubits,
            lengths=length_vector,
            num_samples=num_samples,
            seed=seed,
        )
    except OSError:
        skip_msg = "Skipping tests because tables are missing"
        raise NotImplementedError(skip_msg)  # pylint: disable=raise-missing-from
    return rb_exp.circuits()


class RandomizedBenchmarkingBenchmark:
    # parameters for RB (1&2 qubits):
    params = (
        [
            [0],  # Single qubit RB
            [0, 1],  # Two qubit RB
        ],
    )
    param_names = ["qubits"]
    version = "0.3.0"
    timeout = 600

    def setup(self, qubits):
        length_vector = np.arange(1, 200, 4)
        num_samples = 1
        self.seed = 10
        self.circuits = build_rb_circuit(
            qubits=qubits, length_vector=length_vector, num_samples=num_samples, seed=self.seed
        )

    def teardown(self, _):
        os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

    def time_ibmq_backend_transpile(self, __):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        transpile(
            self.circuits,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            optimization_level=0,
            **{TRANSPILER_SEED_KEYWORD: self.seed},
        )

    def time_ibmq_backend_transpile_single_thread(self, __):
        os.environ["QISKIT_IN_PARALLEL"] = "TRUE"

        # Run with ibmq_16_melbourne configuration
        coupling_map = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        transpile(
            self.circuits,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            optimization_level=0,
            **{TRANSPILER_SEED_KEYWORD: self.seed},
        )
