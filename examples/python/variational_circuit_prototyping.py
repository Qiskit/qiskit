# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Example showing how to quickly prototype variational circuits using NLocal class.
This example shows the most common workflows cases.

Note: Remember to set the values and perform simulations or actual quantum computations based on your specific task.

Variational or parametrized quantum circuits are quantum algorithms characterized by their reliance on freely 
adjustable parameters. Similar to conventional quantum circuits, 
they comprise 3 essential elements:
"""
from qiskit import QuantumCircuit, Parameter
from qiskit.circuit.library import NLocal

"""
1. Preparation of a specific initial quantum state:

You can create an initial state QuantumCircuit and prepend it to the NLocal circuit.
"""
initial_state = QuantumCircuit(3)

# Define your initial state preparation here
nlocal_circuit = NLocal(3, rotation_blocks=None, entanglement_blocks=None, initial_state=initial_state)

"""
2. A quantum circuit U(θ) parameterized by a set of free parameters θ:

You can specify the rotation and entanglement blocks that act on the circuit. 
These blocks can contain parameters that you can adjust later.
"""
rotation_blocks = QuantumCircuit(3)

# Add parameterized gates to the rotation_blocks
theta = Parameter('θ')
rotation_blocks.ry(theta, 0)
entanglement_blocks = QuantumCircuit(3)

# Add gates to the entanglement_blocks
nlocal_circuit = NLocal(3, rotation_blocks, entanglement_blocks)

"""
3. Measurement of an observable ^B at the output:

You can add measurement gates or observable gates to measure the final state of the circuit. 
Here's an example of adding measurements to all qubits.
"""
for qubit in range(3):
    nlocal_circuit.measure(qubit, qubit)
