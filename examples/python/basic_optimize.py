# coding=utf-8
"""Demo basic optimization: Remove Zero Rotations and Remove Double CNOTs."""

import sys
from qiskit import QuantumProgram
from qiskit import CompositeGate
from qiskit.extensions.quantum_initialiser import Initialiser
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.rx import RXGate
# import Qconfig

sys.path.append(
    "../../")  # solve the relative dependencies if you clone QISKit from the Git repo and use like a global.

Q_SPECS = {
    "name": "Program-tutorial",
    "circuits": [{
        "name": "Circuit",
        "quantum_registers": [{
            "name": "qr",
            "size": 4
        }],
        "classical_registers": [{
            "name": "cr",
            "size": 4
        }]}],
}
Q_program = QuantumProgram(specs=Q_SPECS)
circuit = Q_program.get_circuit("Circuit")
quantum_r = Q_program.get_quantum_register("qr")
classical_r = Q_program.get_classical_register('cr')

circuit.h(quantum_r[0])
circuit.rx(0,quantum_r[0])

circuit.cx(quantum_r[0], quantum_r[1])
circuit.cx(quantum_r[0], quantum_r[1])

circuit.h(quantum_r[0])

circuit.cx(quantum_r[0], quantum_r[1])
composite_gate_1 = CompositeGate("composite1", [], [quantum_r[x] for x in range(4)])
# noinspection PyProtectedMember
composite_gate_1._attach(CnotGate(quantum_r[0], quantum_r[1]))
# noinspection PyProtectedMember
circuit._attach(composite_gate_1)

circuit.h(quantum_r[0])

composite_gate_2 = CompositeGate("composite2", [], [quantum_r[x] for x in range(4)])
# noinspection PyProtectedMember
composite_gate_2._attach(CnotGate(quantum_r[0], quantum_r[1]))
# noinspection PyProtectedMember
circuit._attach(composite_gate_2)
circuit.cx(quantum_r[0], quantum_r[1])

circuit.h(quantum_r[0])

composite_gate_3 = CompositeGate("composite3", [], [quantum_r[x] for x in range(4)])
# noinspection PyProtectedMember
composite_gate_3._attach(CnotGate(quantum_r[0], quantum_r[1]))
# noinspection PyProtectedMember
composite_gate_3._attach(CnotGate(quantum_r[0], quantum_r[2]))
# noinspection PyProtectedMember
circuit._attach(composite_gate_3)

circuit.h(quantum_r[0])

composite_gate_4 = CompositeGate("composite4", [], [quantum_r[x] for x in range(4)])
# noinspection PyProtectedMember
composite_gate_4._attach(CnotGate(quantum_r[0], quantum_r[1]))

composite_gate_4._attach(RXGate(0,quantum_r[0]))

# noinspection PyProtectedMember
composite_gate_4._attach(CnotGate(quantum_r[0], quantum_r[1]))
# noinspection PyProtectedMember
circuit._attach(composite_gate_4)

print("Removed Zero Rotations: " + circuit.remove_zero_rotations().__str__())

print("Removed Double CNOTs: " + circuit.remove_double_cnots_once().__str__())

# QASM from a program

QASM_source = Q_program.get_qasm("Circuit")

print(QASM_source)

#device = 'local_qasm_simulator'  # 'simulator' #Backed where execute your program, in this case in the on line simulator
#circuits = ['Circuit']  # Group of circuits to exec

# Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) #set the APIToken and API url

#Q_program.compile(circuits, device)  # Compile your program
