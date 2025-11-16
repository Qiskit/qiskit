#!/usr/bin/env python3
"""Standalone performance test for circuit_to_dag optimization fix.

This script tests the performance improvement for issue #15281 where
circuit_to_dag conversion with qubit_order/clbit_order parameters was
significantly slower.
"""

import time
import sys

# Add the qiskit package to the path
sys.path.insert(0, '.')

try:
    from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
    from qiskit.converters import circuit_to_dag
except ImportError as e:
    print(f"Error importing Qiskit: {e}")
    print("Make sure Qiskit is installed or the package is built.")
    sys.exit(1)


def test_performance():
    """Test that circuit_to_dag conversion with order parameters is performant."""
    print("Testing circuit_to_dag performance with qubit_order/clbit_order...")
    print("=" * 70)
    
    # Create a moderately sized circuit to test performance
    num_qubits = 100
    num_clbits = 100
    qr = QuantumRegister(num_qubits, "qr")
    cr = ClassicalRegister(num_clbits, "cr")
    
    qc = QuantumCircuit(qr, cr)
    # Add some operations to make the circuit non-trivial
    for i in range(min(num_qubits, num_clbits)):
        qc.h(qr[i])
        qc.measure(qr[i], cr[i])
    
    print(f"Created circuit with {num_qubits} qubits, {num_clbits} clbits, "
          f"and {len(qc)} operations")
    
    # Test conversion without order (baseline)
    print("\n1. Testing conversion WITHOUT qubit_order/clbit_order (baseline)...")
    start = time.perf_counter()
    dag_no_order = circuit_to_dag(qc)
    time_no_order = time.perf_counter() - start
    print(f"   Time: {time_no_order:.6f} seconds")
    
    # Test conversion with order (the optimized path)
    print("\n2. Testing conversion WITH qubit_order/clbit_order (optimized path)...")
    qubits_permuted = list(reversed(qc.qubits))
    clbits_permuted = list(reversed(qc.clbits))
    
    start = time.perf_counter()
    dag_with_order = circuit_to_dag(qc, qubit_order=qubits_permuted, clbit_order=clbits_permuted)
    time_with_order = time.perf_counter() - start
    print(f"   Time: {time_with_order:.6f} seconds")
    
    # Verify correctness
    print("\n3. Verifying correctness...")
    assert len(dag_no_order.qubits) == num_qubits, "Qubit count mismatch (no order)"
    assert len(dag_with_order.qubits) == num_qubits, "Qubit count mismatch (with order)"
    assert len(dag_no_order.clbits) == num_clbits, "Clbit count mismatch (no order)"
    assert len(dag_with_order.clbits) == num_clbits, "Clbit count mismatch (with order)"
    assert list(dag_with_order.qubits) == qubits_permuted, "Qubit order mismatch"
    assert list(dag_with_order.clbits) == clbits_permuted, "Clbit order mismatch"
    print("   ✓ Correctness checks passed")
    
    # Performance analysis
    print("\n4. Performance analysis...")
    speedup_ratio = time_no_order / time_with_order if time_with_order > 0 else float('inf')
    slowdown_ratio = time_with_order / time_no_order if time_no_order > 0 else float('inf')
    
    print(f"   Baseline (no order):     {time_no_order:.6f}s")
    print(f"   With order:              {time_with_order:.6f}s")
    print(f"   Ratio (with/baseline):   {slowdown_ratio:.2f}x")
    
    # Performance check: conversion with order should not be significantly slower
    # The original regression was 193x slower, so we allow up to 3x overhead
    max_allowed_slowdown = 3.0
    if slowdown_ratio <= max_allowed_slowdown:
        print(f"\n✓ SUCCESS: Conversion with order is within acceptable performance bounds")
        print(f"  (allowed: up to {max_allowed_slowdown}x slower, actual: {slowdown_ratio:.2f}x)")
        return True
    else:
        print(f"\n✗ FAILURE: Conversion with order is too slow!")
        print(f"  (allowed: up to {max_allowed_slowdown}x slower, actual: {slowdown_ratio:.2f}x)")
        print(f"  This indicates a performance regression.")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Performance Test for circuit_to_dag Optimization (Issue #15281)")
    print("=" * 70)
    
    success = test_performance()
    
    print("\n" + "=" * 70)
    if success:
        print("Test PASSED: Performance is acceptable")
        sys.exit(0)
    else:
        print("Test FAILED: Performance regression detected")
        sys.exit(1)

