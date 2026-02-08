
import gzip
import io
import json
import random
import tempfile
import unittest
import warnings
import re
from qiskit_aer import AerSimulator

import ddt
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CASE_DEFAULT, IfElseOp, WhileLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr, types
from qiskit.circuit import Clbit
from qiskit.circuit import Qubit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import (
    XGate,
    ZGate,
    CXGate,
    RYGate,
    QFT,
    QFTGate,
    QAOAAnsatz,
    PauliEvolutionGate,
    DCXGate,
    MCU1Gate,
    MCXGate,
    MCXGrayCode,
    MCXRecursive,
    MCXVChain,
    MCMTGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
    UnitaryGate,
    DiagonalGate,
    PauliFeatureMap,
    ZZFeatureMap,
    RealAmplitudes,
    pauli_feature_map,
    zz_feature_map,
    qaoa_ansatz,
    real_amplitudes,
)
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.qpy import dump, load, UnsupportedFeatureForVersion, QPY_COMPATIBILITY_VERSION
from qiskit.quantum_info import Pauli, SparsePauliOp, Clifford
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.utils import optionals
from qiskit import transpile
from test import QiskitTestCase  # pylint: disable=wrong-import-order

def test_1():
    cost_operator = Pauli("ZIIZ")
    qaoa = QAOAAnsatz(cost_operator, reps=2)
    qpy_file = io.BytesIO()
    qaoa.data
    dump(qaoa, qpy_file)
    qpy_file.seek(0)
    new_circ = load(qpy_file)[0]
    print(qaoa)
    print(new_circ)

def test_2():
    qc = QuantumCircuit(2, 1)
    with qc.for_loop(range(5)) as i:
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        with qc.if_test((0, True)):
            qc.break_loop()
    transpile(qc)
    

    # qc = QuantumCircuit(1, 1)

    # with qc.for_loop(range(1000)):
    #     qc.h(0)
    #     qc.measure(0, 0)
    #     with qc.if_test((0, False)):
    #         qc.continue_loop()
    #     qc.break_loop()

    # transpiled = transpile(qc, backend)
    # result = backend.run(transpiled, method=method, shots=100).result()
    
test_2()