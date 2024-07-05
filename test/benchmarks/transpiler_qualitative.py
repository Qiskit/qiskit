# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
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

import os

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import Fake27QPulseV1


class TranspilerQualitativeBench:
    params = ([0, 1, 2, 3], ["stochastic", "sabre"], ["dense", "sabre"])
    param_names = ["optimization level", "routing method", "layout method"]
    timeout = 600

    # pylint: disable=unused-argument
    def setup(self, optimization_level, routing_method, layout_method):
        self.backend = Fake27QPulseV1()
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))

        self.depth_4gt10_v1_81 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_4gt10-v1_81.qasm")
        )
        self.depth_4mod5_v0_19 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_4mod5-v0_19.qasm")
        )
        self.depth_mod8_10_178 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_mod8-10_178.qasm")
        )

        self.time_cnt3_5_179 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_cnt3-5_179.qasm")
        )
        self.time_cnt3_5_180 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_cnt3-5_180.qasm")
        )
        self.time_qft_16 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_qft_16.qasm")
        )

    def track_depth_transpile_4gt10_v1_81(self, optimization_level, routing_method, layout_method):
        return transpile(
            self.depth_4gt10_v1_81,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def track_depth_transpile_4mod5_v0_19(self, optimization_level, routing_method, layout_method):
        return transpile(
            self.depth_4mod5_v0_19,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def track_depth_transpile_mod8_10_178(self, optimization_level, routing_method, layout_method):
        return transpile(
            self.depth_mod8_10_178,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def time_transpile_time_cnt3_5_179(self, optimization_level, routing_method, layout_method):
        transpile(
            self.time_cnt3_5_179,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )

    def time_transpile_time_cnt3_5_180(self, optimization_level, routing_method, layout_method):
        transpile(
            self.time_cnt3_5_180,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )

    def time_transpile_time_qft_16(self, optimization_level, routing_method, layout_method):
        transpile(
            self.time_qft_16,
            self.backend,
            routing_method=routing_method,
            layout_method=layout_method,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )
