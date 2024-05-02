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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.transpiler import PassManager
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    SdgGate,
    SGate,
    SXGate,
    SXdgGate,
    SwapGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
    LinearFunction,
)
class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.sdg(0)
        qc.h(0)
        self.definition = qc

class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.append(VGate(), [q[0]], [])
        qc.append(VGate(), [q[0]], [])
        self.definition = qc

class HLSPluginsSuite:

    def _construct_linear_circuit(self, num_qubits: int):
        """Construct linear circuit."""
        qc = QuantumCircuit(num_qubits)
        for i in range(1, num_qubits):
            qc.cx(i - 1, i)
        return qc

    def random_clifford_circuit(self, num_qubits, num_gates, gates="all", seed=None):
        """Generate a pseudo random Clifford circuit."""

        qubits_1_gates = ["i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "v", "w"]
        qubits_2_gates = ["cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"]
        if gates == "all":
            if num_qubits == 1:
                gates = qubits_1_gates
            else:
                gates = qubits_1_gates + qubits_2_gates

        instructions = {
            "i": (IGate(), 1),
            "x": (XGate(), 1),
            "y": (YGate(), 1),
            "z": (ZGate(), 1),
            "h": (HGate(), 1),
            "s": (SGate(), 1),
            "sdg": (SdgGate(), 1),
            "sx": (SXGate(), 1),
            "sxdg": (SXdgGate(), 1),
            "v": (VGate(), 1),
            "w": (WGate(), 1),
            "cx": (CXGate(), 2),
            "cy": (CYGate(), 2),
            "cz": (CZGate(), 2),
            "swap": (SwapGate(), 2),
            "iswap": (iSwapGate(), 2),
            "ecr": (ECRGate(), 2),
            "dcx": (DCXGate(), 2),
        }

        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        samples = rng.choice(gates, num_gates)

        circ = QuantumCircuit(num_qubits)

        for name in samples:
            gate, nqargs = instructions[name]
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            circ.append(gate, qargs)

        return circ

    def setUp(self):
        num_qubits = 100
        rng = np.random.default_rng(1234)
        self.qc_clifford = self.random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)

        linear_function = LinearFunction(self._construct_linear_circuit(num_qubits))
        self.qc_linear = QuantumCircuit(num_qubits)
        self.qc_linear.append(linear_function, list(range(num_qubits)))

    # Clifford Synthesis Plugins
    def time_synth_ag(self):
        """Test A&G synthesis for set of {num_qubits}-qubit Cliffords"""
        hls_config = HLSConfig(clifford=["ag"])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_clifford)

    def time_synth_bm(self):
        """Test B&M synthesis for set of {num_qubits}-qubit Cliffords"""
        hls_config = HLSConfig(clifford=["bm"])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_clifford)

    def time_synth_greedy(self):
        """Test B&M synthesis for set of {num_qubits}-qubit Cliffords"""
        hls_config = HLSConfig(clifford=["greedy"])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_clifford)

    def time_synth_full(self):
        """Test synthesis for set of {num_qubits}-qubit Cliffords"""
        hls_config = HLSConfig(clifford=["full"])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_clifford)

    # Linear Synthesis Plugins
    def time_pmh_linear_func_plugin(self):
        hls_config = HLSConfig(linear_function=[("pmh", {})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_linear)

    def time_kms_linear_func_plugin(self):
        hls_config = HLSConfig(linear_function=[("kms", {})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        _ = pm.run(self.qc_linear)