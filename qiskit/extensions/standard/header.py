"""
Standard extension's OPENQASM header update.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit


if not hasattr(QuantumCircuit, '_extension_standard'):
    QuantumCircuit._extension_standard = True
    QuantumCircuit.header = QuantumCircuit.header + "\n" \
        + "include \"qelib1.inc\";"
