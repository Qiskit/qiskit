import numpy as np
from plum import dispatch
from typing import Union

from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.quantum_info import Operator
from qiskit.result.counts import Counts

from qiskit._accelerate import pauli_expval

# TODO: I am still working on how to add doc strings in the appropriate places.


@dispatch
def _to_rust_array(state: Statevector):
    return state.data


@dispatch
def _to_rust_array(state: DensityMatrix):
    return np.ravel(state.data, order="F")


@dispatch
def expval_pauli(state: Statevector, z_mask):
    return pauli_expval.expval_pauli_no_x(
        _to_rust_array(state), state.num_qubits, z_mask
    )


@dispatch
def expval_pauli(state: Statevector, z_mask, x_mask, y_phase, x_max):
    return pauli_expval.expval_pauli_with_x(
        _to_rust_array(state), state.num_qubits, z_mask, x_mask, y_phase, x_max
    )


@dispatch
def expval_pauli(state: DensityMatrix, z_mask):
    data = _to_rust_array(state)
    return pauli_expval.density_expval_pauli_no_x(data, state.num_qubits, z_mask)


@dispatch
def expval_pauli(state: DensityMatrix, z_mask, x_mask, y_phase, x_max):
    data = _to_rust_array(state)
    return pauli_expval.density_expval_pauli_with_x(
        data, state.num_qubits, z_mask, x_mask, y_phase, x_max
    )


@dispatch
def sum_of_probs(state: Statevector):
    return np.linalg.norm(state.data)


@dispatch
def sum_of_probs(state: DensityMatrix):
    return state.trace()


# This will also be used in methods for Cliffords
# @dispatch  # We are not actually using MD here
def pauli_phase(pauli: Pauli):
    return (-1j) ** pauli.phase if pauli.phase else 1


## Function `expectation_value`.
## expectation_value(oper, state)
## expectation_value(oper, state, qargs) # to restrict to a subset of qubits
## expectation_value(state) # if the operator is implied
## expectation_value(state, qargs) # operator implied, restricted subset of qubits

## Compare this to the implementations in quantum_info classes.
## Here, we have a single method for Statevector, DensityMatrix rather than two.
## Furthermore, the internal structure of `state` is not accessed in this method.
@dispatch (precedence=1)
def expectation_value(pauli: Pauli, state: Union[Statevector, DensityMatrix], qargs: Union[list, range]):

    qubits = np.array(qargs)
    x_mask = np.dot(1 << qubits, pauli.x)
    z_mask = np.dot(1 << qubits, pauli.z)
    _pauli_phase = pauli_phase(pauli)

    if x_mask + z_mask == 0:
        return _pauli_phase * sum_of_probs(state)

    if x_mask == 0:
        return _pauli_phase * expval_pauli(state, z_mask)

    x_max = qubits[pauli.x][-1]
    y_phase = (-1j) ** pauli._count_y()
    return _pauli_phase * expval_pauli(
        state, z_mask, x_mask, y_phase, x_max
    )


@dispatch
def expectation_value(pauli: Pauli, state: Union[Statevector, DensityMatrix]):
    return expectation_value(pauli, state, range(len(pauli)))


@dispatch
def expectation_value(oper: SparsePauliOp, state: Union[Statevector, DensityMatrix], qargs):
    return sum(
        coeff * expectation_value_pauli(Pauli((z, x)), qargs)
        for z, x, coeff in zip(oper.paulis.z, oper.paulis.x, oper.coeffs)
    )


@dispatch
def expectation_value(oper, state: Statevector, qargs: list):
    val = state.evolve(oper, qargs=qargs)
    conj = state.conjugate()
    return np.dot(conj.data, val.data)


@dispatch
def expectation_value(oper: Operator, state: DensityMatrix, qargs: list):
    return np.trace(Operator(state).dot(oper, qargs=qargs).data)


@dispatch
def expectation_value(oper, state: DensityMatrix, qargs: list):
    return expectation_value(Operator(oper), state, qargs)


# Compute the expectation value of Counts in the same basis that was measured
# to collect the counts.
@dispatch
def expectation_value(counts: Counts):
    _sum = 0
    total_counts = 0
    for bitstr, _count in counts.items():
        _sum += _count * (-1) ** bitstr.count('1')
        total_counts += _count
    return _sum / total_counts


# Compute the expectation value of Counts in the same basis that was measured
# to collect the counts, for a subset of the qubits.
# `op` is a binary string specifying which qubits
# to include.
# For example `1100` specifies computing the expectation value of the
# first two qubits in a four-qubit register.
#
# This is not the most convenient/efficient way to specify these selected qubits.
# A more convenient way to pass this info should be implemented.
@dispatch
def expectation_value(counts: Counts, qargs: str):
    _sum = 0
    total_counts = 0
    mask = int(qargs, base=2)
    for bitstr, _count in counts.items():
        parity = bin(int(bitstr, base=2) & mask).count('1')
        _sum += _count * (-1) ** parity
        total_counts += _count
    return _sum / total_counts
