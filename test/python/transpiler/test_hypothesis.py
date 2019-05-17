# Construct an n-qubit, m-clbit circuit
# Compile at each optimization level, for each device (with more than n-qubits)
# Simulate and verify results of every transpiled circuit match that of initial circuit

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

oneQ_gates = [ HGate, IdGate, SGate, SdgGate, TGate, TdgGate, U0Gate, XGate, YGate, ZGate, Reset ]
twoQ_gates = [ CnotGate, CyGate, CzGate, SwapGate, CHGate ]
threeQ_gates = [ ToffoliGate, FredkinGate ]

oneQ_oneP_gates = [ U1Gate, RXGate, RYGate, RZGate ]
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

    @precondition(lambda self: len(self.qc.qubits) <= 5)
    @rule(target=qubits, n=st.integers(min_value=1, max_value=5))
    def add_qreg(self, n):
        n = max(n, 5 - len(self.qc.qubits))
        qreg = QuantumRegister(n)
        self.qc.add_register(qreg)
        return multiple(list(qreg))

    @rule(target=clbits, n=st.integers(1, 5))
    def add_creg(self, n):
        creg = ClassicalRegister(n)
        self.qc.add_register(creg)
        return multiple(list(creg))

    ### Gates of various shapes

    #@precondition(_at_least_n_qubits(self, 1))
    @rule(gate=st.one_of(oneQ_gates), qarg=st.lists(qubits))
    def add_1q_gate(self, gate, qarg):
        self.qc.append(gate(), [qarg], [])
    
    # @precondition(lambda self: len(self.qc.qubits) > 1)
    # @rule(gate=st.sampled_from(2q_gates), ctrl=st.integers(), tgt=st.integers())
    # def add_2q_gate(self, gate, ctrl, tgt):
    #     qubit_count = len(self.qc.qubits)
    #     if ctrl % qubit_count != tgt % qubit_count:
    #         self.qc.append(gate(), ctrl % qubit_count, tgt % qubit_count)

    # def add_3q_gate():
    #     pass

    # def add_1q1p_gate():
    #     pass

    # @precondition(lambda self: len(self.qc.qubits) > 1 and len(self.qc.clbits) > 1)
    # @rule(qubit=st.integers(), clbit=st.integers())
    # def add_1q1c_gate(self, qubit, cbit):
    #     self.qc.measure(qubit % len(self.qc.qubits), clbit % len(self.qc.clbits))

    # @precondition(_at_least_n_qubits(self, 1))
    # @rule(gate=st.sampled_from(variadic_gates, data=st.data())
    # def add_variQ_gate(self, gate, data):
    #     qubits = data.draw(st.lists(self.qc.qubits, min_size=1, unique=True)
    #     self.qc.append(gate(), qubits, [])

    # Properties to check

    @invariant()
    def qasm(self):
        self.qc.qasm()

    # Could be an invariant, but then it's run at ~every step
    @precondition(lambda self: len(self.qc.qubits) > 1)
    @rule()
    def xpile(self):
        aer_qasm_simulator = Aer.get_backend('qasm_simulator')
        basicaer_qasm_simulator = Aer.get_backend('qasm_simulator')

        # If measure/reset => qasm => keys eq, values np.allclose
        # Else => statevecector? np.allclose
        execute(self.qc, backend = aer_qasm_simulator).result().get_counts()

        levels = [0,1,2,3]
        backends = [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo()]
        
        assertDictAlmostEqual(None, {}, {})
        
        
    


TestQuantumCircuit = QCircuitMachine.TestCase
