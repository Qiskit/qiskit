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

"""Module for estimating randomized benchmarking."""

import numpy as np
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit.providers.basicaer import QasmSimulatorPy

try:
    from qiskit.compiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_transpiler'
except ImportError:
    from qiskit.transpiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_mapper'


def build_rb_circuit(nseeds=1, length_vector=None,
                     rb_pattern=None, length_multiplier=1,
                     seed_offset=0, align_cliffs=False, seed=None):
    """
    Randomized Benchmarking sequences.
    """
    if not seed:
        np.random.seed(10)
    else:
        np.random.seed(seed)
    rb_opts = {}
    rb_opts['nseeds'] = nseeds
    rb_opts['length_vector'] = length_vector
    rb_opts['rb_pattern'] = rb_pattern
    rb_opts['length_multiplier'] = length_multiplier
    rb_opts['seed_offset'] = seed_offset
    rb_opts['align_cliffs'] = align_cliffs

    # Generate the sequences
    try:
        rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
    except OSError:
        skip_msg = ('Skipping tests because '
                    'tables are missing')
        raise NotImplementedError(skip_msg)
    all_circuits = []
    for seq in rb_circs:
        all_circuits += seq
    return all_circuits


class RandomizedBenchmarkingBenchmark:
    # parameters for RB (1&2 qubits):
    params = ([[[0]], [[0, 1]], [[0, 2], [1]]],)
    param_names = ['rb_pattern']
    version = '0.1.1'
    timeout = 600

    def setup(self, rb_pattern):
        length_vector = np.arange(1, 200, 4)
        nseeds = 1
        self.seed = 10
        self.circuits = build_rb_circuit(nseeds=nseeds,
                                         length_vector=length_vector,
                                         rb_pattern=rb_pattern,
                                         seed=self.seed)
        self.sim_backend = QasmSimulatorPy()

    def time_simulator_transpile(self, __):
        transpile(self.circuits, self.sim_backend,
                  **{TRANSPILER_SEED_KEYWORD: self.seed})

    def time_ibmq_backend_transpile(self, __):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                        [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                        [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                        [13, 12]]

        transpile(self.circuits,
                  basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                  coupling_map=coupling_map,
                  **{TRANSPILER_SEED_KEYWORD: self.seed})
