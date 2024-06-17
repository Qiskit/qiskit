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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import os
import itertools

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, QuantumVolume
from .utils import dtc_unitary, multi_control_circuit


def build_circuit(width, gates):
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            qc.h(qr[k])
        for k in range(width - 1):
            qc.cx(qr[k], qr[k + 1])

    return qc


class CircuitConstructionBench:
    params = ([1, 2, 5, 8, 14, 20], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates"]
    timeout = 600

    def setup(self, width, gates):
        self.empty_circuit = build_circuit(width, 0)
        self.sample_circuit = build_circuit(width, gates)

    def time_circuit_construction(self, width, gates):
        build_circuit(width, gates)

    def time_circuit_extend(self, _, __):
        self.empty_circuit.compose(self.sample_circuit, inplace=True)

    def time_circuit_copy(self, _, __):
        self.sample_circuit.copy()


def build_parameterized_circuit(width, gates, param_count):
    params = [Parameter("param-%s" % x) for x in range(param_count)]
    param_iter = itertools.cycle(params)

    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            param = next(param_iter)
            qc.r(0, param, qr[k])
        for k in range(width - 1):
            param = next(param_iter)
            qc.crx(param, qr[k], qr[k + 1])

    return qc, params


class ParameterizedCircuitConstructionBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, _, gates, params):
        if params > gates:
            raise NotImplementedError

    def time_build_parameterized_circuit(self, width, gates, params):
        build_parameterized_circuit(width, gates, params)


class ParameterizedCircuitBindBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, width, gates, params):
        if params > gates:
            raise NotImplementedError
        self.circuit, self.params = build_parameterized_circuit(width, gates, params)

    def time_bind_params(self, _, __, ___):
        # TODO: write more complete benchmarks of assign_parameters
        #  that test more of the input formats / combinations
        self.circuit.assign_parameters({x: 3.14 for x in self.params})


class ParamaterizedDifferentCircuit:
    SEED = 12345
    ITER_CIRCUIT_WIDTH = 16
    N = 100

    def time_QV100_build(self):
        """Measures an SDKs ability to build a 100Q
        QV circit from scratch.
        """
        return QuantumVolume(100, 100, self.SEED)

    def time_multi_control_circuit(self):
        """Measures an SDKs ability to build a circuit
        with a multi-controlled X-gate
        """
        out = multi_control_circuit(self.ITER_CIRCUIT_WIDTH)
        return out

class GateCountCircuit:
    SEED = 12345
    ITER_CIRCUIT_WIDTH = 16
    N = 100

    def track_DTC100_set_build(self):
        """Measures an SDKs ability to build a set
        of 100Q DTC circuits out to 100 layers of
        the underlying unitary
        """
        max_cycles = 100
        num_qubits = 100
        initial_state = QuantumCircuit(num_qubits)
        dtc_circuit = dtc_unitary(num_qubits, g=0.95, seed=self.SEED)

        circs = [initial_state]
        for tt in range(max_cycles):
            qc = circs[tt].compose(dtc_circuit)
            circs.append(qc)
            result = circs[-1]

        assert result.num_parameters == 0
        return result.count_ops()

    def time_param_circSU2_100_build(self):
        """Measures an SDKs ability to build a
        parameterized efficient SU2 circuit with circular entanglement
        over 100Q utilizing 4 repetitions.  This will yield a
        circuit with 1000 parameters
        """
        out = EfficientSU2(self.N, reps=4, entanglement="circular", flatten=True)
        out._build()
        assert out.num_parameters == 1000
        return out
    
    def time_param_circSU2_100_bind(self):
        """Measures an SDKs ability to build a
        parameterized efficient SU2 circuit with circular entanglement
        over 100Q utilizing 4 repetitions.  This will yield a
        circuit with 1000 parameters
        """
        out = EfficientSU2(self.N, reps=4, entanglement="circular", flatten=True)
        out._build()
        assert out.num_parameters == 0
        return out

    def track_QV100_qasm2_import(self):
        """QASM import of QV100 circuit"""
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))

        out = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, "qv_N100_12345.qasm"))
        ops = out.count_ops()
        assert ops.get("rz", 0) == 120000
        assert ops.get("rx", 0) == 80000
        assert ops.get("cx", 0) == 15000
        return ops
