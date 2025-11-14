#!/usr/bin/env python3
"""Benchmark BasicSimulator performance for Clifford and non-Clifford circuits."""

import time
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

print("=" * 70)
print("BasicSimulator Performance Benchmark - Clifford vs Non-Clifford")
print("=" * 70)

# Test with different qubit counts
for n_qubits in [10, 15, 20]:
    print(f"\n{'=' * 70}")
    print(f"Testing with {n_qubits} qubits, 1000 shots")
    print(f"{'=' * 70}")
    
    # Create Clifford circuit (only H, CNOT gates)
    clifford_qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        clifford_qc.h(i)
    for i in range(n_qubits-1):
        clifford_qc.cx(i, i+1)
    clifford_qc.measure(range(n_qubits), range(n_qubits))
    
    # Create non-Clifford circuit (with T gates)
    non_clifford_qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        non_clifford_qc.h(i)
        non_clifford_qc.t(i)  # T gate makes it non-Clifford
    for i in range(n_qubits-1):
        non_clifford_qc.cx(i, i+1)
    non_clifford_qc.measure(range(n_qubits), range(n_qubits))
    
    # Test 1: Non-Clifford with default backend
    print("\n1. Non-Clifford circuit (default, no opt-in):")
    backend_default = BasicSimulator()
    start = time.time()
    job = backend_default.run(non_clifford_qc, shots=1000)
    result = job.result()
    non_clifford_time = time.time() - start
    print(f"   Time: {non_clifford_time:.3f}s")
    
    # Test 2: Clifford without optimization (opt-in OFF)
    print("\n2. Clifford circuit (optimization OFF, default):")
    start = time.time()
    job = backend_default.run(clifford_qc, shots=1000)
    result = job.result()
    clifford_default_time = time.time() - start
    print(f"   Time: {clifford_default_time:.3f}s")
    
    # Test 3: Clifford WITH optimization (opt-in ON)
    print("\n3. Clifford circuit (optimization ON, use_clifford_optimization=True):")
    backend_optin = BasicSimulator(use_clifford_optimization=True)
    start = time.time()
    job = backend_optin.run(clifford_qc, shots=1000)
    result = job.result()
    clifford_optin_time = time.time() - start
    print(f"   Time: {clifford_optin_time:.3f}s")
    
    # Calculate speedup
    speedup = clifford_default_time / clifford_optin_time
    print(f"\n   Speedup with optimization: {speedup:.2f}x")
    print(f"   Non-Clifford time unchanged: {non_clifford_time:.3f}s (not using optimization)")

print("\n" + "=" * 70)
print("Benchmark Complete")
print("=" * 70)
