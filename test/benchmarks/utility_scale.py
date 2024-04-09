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

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object

import os

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit import qasm2


class UtilityScaleBenchmarks:
    params = ["cx", "cz", "ecr"]
    param_names = ["2q gate"]

    def setup(self, basis_gate):
        cmap = CouplingMap.from_heavy_hex(9)
        basis_gates = ["rz", "x", "sx", basis_gate, "id"]
        backend = GenericBackendV2(
            cmap.size(), basis_gates, coupling_map=cmap, control_flow=True, seed=12345678942
        )
        self.pm = generate_preset_pass_manager(2, backend, seed_transpiler=1234567845)
        qasm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm")
        self.qft_qasm = os.path.join(qasm_dir, "qft_N100.qasm")
        self.qft_qc = QuantumCircuit.from_qasm_file(self.qft_qasm)
        self.square_heisenberg_qasm = os.path.join(qasm_dir, "square_heisenberg_N100.qasm")
        self.square_heisenberg_qc = QuantumCircuit.from_qasm_file(self.square_heisenberg_qasm)
        self.qaoa_qasm = os.path.join(qasm_dir, "qaoa_barabasi_albert_N100_3reps.qasm")
        self.qaoa_qc = QuantumCircuit.from_qasm_file(self.qaoa_qasm)

    def time_parse_qft_n100(self, _):
        qasm2.load(
            self.qft_qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    def time_parse_square_heisenberg_n100(self, _):
        qasm2.load(
            self.square_heisenberg_qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    def time_parse_qaoa_n100(self, _):
        qasm2.load(
            self.qaoa_qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    def time_qft(self, _):
        self.pm.run(self.qft_qc)

    def track_qft_depth(self, basis_gate):
        res = self.pm.run(self.qft_qc)
        return res.depth(filter_function=lambda x: x.operation.name == basis_gate)

    def time_square_heisenberg(self, _):
        self.pm.run(self.square_heisenberg_qc)

    def track_square_heisenberg_depth(self, basis_gate):
        res = self.pm.run(self.square_heisenberg_qc)
        return res.depth(filter_function=lambda x: x.operation.name == basis_gate)

    def time_qaoa(self, _):
        self.pm.run(self.qaoa_qc)

    def track_qaoa_depth(self, basis_gate):
        res = self.pm.run(self.qaoa_qc)
        return res.depth(filter_function=lambda x: x.operation.name == basis_gate)
