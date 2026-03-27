from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy import sparse

try:
    # A 0-qubit operator (just a scalar identity coeff)
    # Correct Syntax: Labels can't be empty for 0 qubits? Wait.
    # Qiskit SparsePauliOp([]) for 0 qubits?
    # Or labels=[""] with num_qubits=0?
    op = SparsePauliOp.from_list([("", 1.0)], num_qubits=0)
    
    # This calls the Rust `to_matrix_sparse` function internally
    print("Attempting to_matrix(sparse=True) on 0-qubit operator...")
    sparse_mat = op.to_matrix(sparse=True)
    
    print("\nSuccess! Matrix shape:", sparse_mat.shape)
    print("Matrix Entries:\n", sparse_mat.toarray())
except Exception as e:
    print("\nFAILURE! Error encountered:", str(e))
