from qiskit import QuantumCircuit
from qiskit.circuit.library import GlobalPhaseGate
import numpy as np

# Build the circuit
circ = QuantumCircuit(5)
circ.h(0)
circ.cx(3, 2)
circ.append(GlobalPhaseGate(np.pi))
circ.append(GlobalPhaseGate(5 * np.pi))
circ.x(0)

# 1. Test the Text Drawer
print("=== TEXT DRAWER OUTPUT ===")
print(circ.draw('text'))

# 2. Test the Matplotlib Drawer
print("\nSaving MPL output to mpl_bug.png...")
circ.draw('mpl', filename='mpl_bug.png')
