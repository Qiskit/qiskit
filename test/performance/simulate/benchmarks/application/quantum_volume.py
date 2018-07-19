"""
Generate (or run) randomized circuits for Quantum Volume analysis.

Example run:
  python quantum_volume.py -n 5 -d 5
"""

import math
from numpy import random
from scipy import linalg
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.mapper import two_qubit_kak


class QuantumVolume:
    """
    QuantumVolume Generator
    """
    def __init__(self, seed):
        self.name = "qv"
        self.seed = seed

    @classmethod
    def random_su(cls, num):
        """Return an n x n Haar distributed unitary matrix,
        using QR-decomposition on a random n x n.
        """
        x_value = (random.randn(num, num) + 1j * random.randn(num, num))
        q_value, _ = linalg.qr(x_value)           # Q is a unitary matrix
        q_value /= pow(linalg.det(q_value), 1/num)  # make Q a special unitary
        return q_value

    @classmethod
    def build_model_circuit(cls, num, depth):
        """Create a quantum program containing model circuits.
        The model circuits consist of layers of Haar random
        elements of SU(4) applied between corresponding pairs
        of qubits in a random bipartition.

        Args:
            num (int): number of qubits
            depth (int): ideal depth of each model circuit (over SU(4))

        Returns:
            QuantumCircuit: quantum volume circuits
        """
        # Create quantum/classical registers of size n

        q_r = QuantumRegister(num)
        c_r = ClassicalRegister(num)
        # For each sample number, build the model circuits
        # Initialize empty circuit
        circuit = QuantumCircuit(q_r, c_r)
        # For each layer
        for _ in range(depth):
            # Generate uniformly random permutation Pj of [0...n-1]
            perm = random.permutation(num)
            # For each consecutive pair in Pj, generate Haar random SU(4)
            # Decompose each SU(4) into CNOT + SU(2) and add to Ci
            for k in range(math.floor(num/2)):
                qubits = [int(perm[2*k]), int(perm[2*k+1])]
                for gate in two_qubit_kak(cls.random_su(4)):
                    i_0 = qubits[gate["args"][0]]
                    if gate["name"] == "cx":
                        i_1 = qubits[gate["args"][1]]
                        circuit.cx(q_r[i_0], q_r[i_1])
                    elif gate["name"] == "u1":
                        circuit.u1(gate["params"][2], q_r[i_0])
                    elif gate["name"] == "u2":
                        circuit.u2(gate["params"][1], gate["params"][2], q_r[i_0])
                    elif gate["name"] == "u3":
                        circuit.u3(gate["params"][0], gate["params"][1],
                                   gate["params"][2], q_r[i_0])
                    elif gate["name"] == "id":
                        pass
        # Barrier before measurement to prevent reordering, then measure
        circuit.barrier(q_r)
        circuit.measure(q_r, c_r)
        # Save sample circuit
        return circuit

    def build_model_circuits(self, num, depth, num_circ=1):
        """
        make circuits
        """
        circuits = []

        for _ in range(num_circ):
            circuit = self.build_model_circuit(num, depth)
            circuits.append(circuit)

        return circuits

    def gen_application(self, app_arg):
        """
        generate application
        """
        random.seed(self.seed)

        qubits = app_arg["qubit"]
        depth = app_arg["depth"]
        circuit = self.build_model_circuit(num=qubits, depth=depth)

        return circuit
