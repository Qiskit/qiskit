# Construct an n-qubit, m-clbit circuit
# Compile at each optimization level, for each device (with more than n-qubits)
# Simulate and verify results of every transpiled circuit match that of initial circuit

import numpy as np

from hypothesis import given, settings, Verbosity
from hypothesis.stateful import multiple, rule, precondition, invariant
from hypothesis.stateful import Bundle, RuleBasedStateMachine

import hypothesis.strategies as st

from qiskit import execute, transpile, Aer, BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Measure, Reset
from qiskit.test.mock import FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo, FakePoughkeepsie
from qiskit.extensions.standard import *

# TBD, conditionals, Parameters

oneQ_gates = [ HGate, IdGate, SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate, Reset ]
twoQ_gates = [ CnotGate, CyGate, CzGate, SwapGate, CHGate ]
threeQ_gates = [ ToffoliGate, FredkinGate ]

oneQ_oneP_gates = [ U0Gate, U1Gate, RXGate, RYGate, RZGate ]
oneQ_twoP_gates = [ U2Gate ]
oneQ_threeP_gates = [ U3Gate ]

twoQ_oneP_gates = [ CrzGate, RZZGate, Cu1Gate ]
twoQ_threeP_gates = [ Cu3Gate ]

oneQ_oneC_gates = [ Measure ]
variadic_gates = [ Barrier ]

backends = [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), FakePoughkeepsie()]

class QCircuitMachine(RuleBasedStateMachine):
    qubits = Bundle('qubits')
    clbits = Bundle('clbits')

    def __init__(self):
        super().__init__()
        self.qc = QuantumCircuit()

    @precondition(lambda self: len(self.qc.qubits) < 5)
    @rule(target=qubits,
          n=st.integers(min_value=1, max_value=5))
    def add_qreg(self, n):
        n = max(n, 5 - len(self.qc.qubits))
        qreg = QuantumRegister(n)
        self.qc.add_register(qreg)
        return multiple(*list(qreg))

    @rule(target=clbits,
          n=st.integers(1, 5))
    def add_creg(self, n):
        creg = ClassicalRegister(n)
        self.qc.add_register(creg)
        return multiple(*list(creg))

    ### Gates of various shapes

    @rule(gate=st.sampled_from(oneQ_gates),
          qarg=qubits)
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

    @rule(gate=st.sampled_from(oneQ_twoP_gates),
          qarg=qubits,
          params=st.lists(
              st.floats(allow_nan=False, allow_infinity=False),
              min_size=2, max_size=2))
    def add_1q2p_gate(self, gate, qarg, params):
        self.qc.append(gate(*params), [qarg])

    @rule(gate=st.sampled_from(oneQ_threeP_gates),
          qarg=qubits,
          params=st.lists(
              st.floats(allow_nan=False, allow_infinity=False),
              min_size=3, max_size=3))
    def add_1q3p_gate(self, gate, qarg, params):
        self.qc.append(gate(*params), [qarg])

    @rule(gate=st.sampled_from(twoQ_oneP_gates),
          qargs=st.lists(qubits, max_size=2, min_size=2, unique=True),
          param=st.floats(allow_nan=False, allow_infinity=False))
    def add_2q1p_gate(self, gate, qargs, param):
        self.qc.append(gate(param), qargs)

    @rule(gate=st.sampled_from(twoQ_threeP_gates),
          qargs=st.lists(qubits, max_size=2, min_size=2, unique=True),
          params=st.lists(
              st.floats(allow_nan=False, allow_infinity=False),
              min_size=3, max_size=3))
    def add_2q3p_gate(self, gate, qargs, params):
        self.qc.append(gate(*params), qargs)

    @rule(gate=st.sampled_from(oneQ_oneC_gates),
          qarg=qubits,
          carg=clbits)
    def add_1q1c_gate(self, gate, qarg, carg):
        self.qc.append(gate(), [qarg], [carg])

    @rule(gate=st.sampled_from(variadic_gates),
          qargs=st.lists(qubits, min_size=1, unique=True))
    def add_variQ_gate(self, gate, qargs):
        self.qc.append(gate(len(qargs)), qargs)

    @precondition(lambda self: len(self.qc.data) > 0)
    @rule(carg=clbits,
          data=st.data())
    def add_c_if_last_gate(self, carg, data):
        creg = carg.register
        val = data.draw(st.integers(min_value=0, max_value=2**len(creg)-1))

        self.qc.data[-1][0].c_if(creg, val)

    # Properties to check

    @invariant()
    def qasm(self):
        # At every step in the process, it should be possible to generate QASM
        self.qc.qasm()

    @precondition(lambda self: any(isinstance(d[0], Measure) for d in self.qc.data))
    @rule(backend=st.one_of(
              st.none(),
              st.sampled_from(backends)),
          opt_level=st.one_of(
              st.none(),
              st.integers(min_value=0, max_value=3)))
    def equivalent_transpile(self, backend, opt_level):
        print('Evaluating circuit at level {} on {}:\n{}'.format(opt_level, backend, self.qc.qasm()))

        aer_qasm_simulator = Aer.get_backend('qasm_simulator')
        aer_counts = execute(self.qc, backend = aer_qasm_simulator).result().get_counts()

        xpiled_qc = transpile(self.qc, backend=backend, optimization_level=opt_level)
        xpiled_aer_counts = execute(xpiled_qc, backend = aer_qasm_simulator).result().get_counts()

        assert counts_equivalent(aer_counts, xpiled_aer_counts), "Counts not equivalent. Original: {} Transpiled: {}".format(aer_counts, xpiled_aer_counts)
        
def counts_equivalent(c1, c2):
    sc1 = np.array(sorted(c1.items(), key=lambda c: c[0]))
    sc2 = np.array(sorted(c2.items(), key=lambda c: c[0]))

    return all(sc1[:,0] == sc2[:,0]) and np.allclose(sc1[:,1].astype(int), sc2[:,1].astype(int))

TestQuantumCircuit = QCircuitMachine.TestCase
