# Construct an n-qubit, m-clbit circuit
# Compile at each optimization level, for each device (with more than n-qubits)
# Simulate and verify results of every transpiled circuit match that of initial circuit

import numpy as np

from hypothesis import given
from hypothesis.stateful import multiple, rule, precondition, invariant, Bundle, RuleBasedStateMachine
import hypothesis.strategies as st

from qiskit.test.mock import FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo

from qiskit import execute, transpile, Aer, BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Measure, Reset

from qiskit.test.base import QiskitTestCase
assertDictAlmostEqual = QiskitTestCase.assertDictAlmostEqual

from qiskit.extensions.standard import *

# TBD, conditionals, Parameters

oneQ_gates = [ HGate, IdGate, SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate, Reset ]
twoQ_gates = [ CnotGate, CyGate, CzGate, SwapGate, CHGate ]
threeQ_gates = [ ToffoliGate, FredkinGate ]

oneQ_oneP_gates = [ U0Gate, U1Gate, RXGate, RYGate, RZGate ]
# oneQ_twoP_gates = [ U2Gate ]
# oneQ_threeP_gates = [ U3Gate ]

# twoQ_oneP_gates = [ CrzGate, RZZGate, Cu1Gate ]
# twoQ_twoP_gates = [ Cu2Gate ]
# twoQ_threeP_gates = [ Cu3Gate ]

oneQ_oneC_gates = [ Measure ]
variadic_gates = [ Barrier ]

class QCircuitMachine(RuleBasedStateMachine):
    qubits = Bundle('qubits')
    clbits = Bundle('clbits')

    def __init__(self):
        super().__init__()
        self.qc = QuantumCircuit()

    @precondition(lambda self: len(self.qc.qubits) < 5)
    @rule(target=qubits, n=st.integers(min_value=1, max_value=5))
    def add_qreg(self, n):
        n = max(n, 5 - len(self.qc.qubits))
        qreg = QuantumRegister(n)
        self.qc.add_register(qreg)
        return multiple(*list(qreg))

    @rule(target=clbits, n=st.integers(1, 5))
    def add_creg(self, n):
        creg = ClassicalRegister(n)
        self.qc.add_register(creg)
        return multiple(*list(creg))

    ### Gates of various shapes

    @rule(gate=st.sampled_from(oneQ_gates), qarg=qubits)
    def add_1q_gate(self, gate, qarg):
        self.qc.append(gate(), [qarg], [])
    
    @rule(gate=st.sampled_from(twoQ_gates),
          qargs=st.lists(qubits, max_size=2, min_size=2, unique=True))
    def add_2q_gate(self, gate, qargs):
        self.qc.append(gate(), qargs)

    @rule(gate=st.sampled_from(threeQ_gates),
          qargs=st.lists(qubits, max_size=3, min_size=3, unique=True))
    def add_3q_gate(self, gate, qargs):
        self.qc.append(gate(), qargs)

    @rule(gate=st.sampled_from(oneQ_oneP_gates),
          qarg=qubits,
          param=st.floats(allow_nan=False, allow_infinity=False))
    def add_1q1p_gate(self, gate, qarg, param):
        self.qc.append(gate(param), [qarg])

    @rule(gate=st.sampled_from(oneQ_oneC_gates), qarg=qubits, carg=clbits)
    def add_1q1c_gate(self, gate, qarg, carg):
        self.qc.append(gate(), [qarg], [carg])

    @rule(gate=st.sampled_from(variadic_gates),
          qargs=st.lists(qubits, min_size=1, unique=True))
    def add_variQ_gate(self, gate, qargs):
        self.qc.append(gate(len(qargs)), qargs)

    # Properties to check

    @invariant()
    def qasm(self):
        # At every step in the process, it should be possible to generate QASM
        self.qc.qasm()

    @precondition(lambda self: any(isinstance(d[0], Measure) for d in self.qc.data))
    @rule()
    def equivalent_transpile(self):
        aer_qasm_simulator = Aer.get_backend('qasm_simulator')
        basicaer_qasm_simulator = Aer.get_backend('qasm_simulator')

        aer_counts = execute(self.qc, backend = aer_qasm_simulator).result().get_counts()
        basicaer_counts = execute(self.qc, backend = aer_qasm_simulator).result().get_counts()

        assert counts_equivalent(aer_counts, basicaer_counts), (aer_counts, basicaer_counts)

        levels = [0,1,2,3]
        backends = [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo()]

        for level in levels:
            for backend in backends:
                xpiled_qc = transpile(self.qc, backend=backend, optimization_level=level)

                xpiled_aer_counts = execute(xpiled_qc, backend = aer_qasm_simulator).result().get_counts()
                xpiled_basicaer_counts = execute(xpiled_qc, backend = basicaer_qasm_simulator).result().get_counts()

                assert counts_equivalent(aer_counts, xpiled_aer_counts)
                assert counts_equivalent(basicaer_counts, xpiled_basicaer_counts)
        
        
def counts_equivalent(c1, c2):
    sc1 = np.array(sorted(c1.items(), key=lambda c: c[0]))
    sc2 = np.array(sorted(c2.items(), key=lambda c: c[0]))

    return all(sc1[:,0] == sc2[:,0]) and np.allclose(sc1[:,1].astype(int), sc2[:,1].astype(int))

TestQuantumCircuit = QCircuitMachine.TestCase
