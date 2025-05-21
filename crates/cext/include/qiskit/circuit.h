// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef QISKIT__CIRCUIT_H
#define QISKIT__CIRCUIT_H

#ifdef QISKIT_C_PYTHON_INTERFACE
#include <Python.h>
#endif

#include "complex.h"
#include "exit.h"
#include <stdlib.h>

/// @ingroup QkCircuit
///
/// Units for circuit delays.
enum QkDelayUnit
#ifdef __cplusplus
    : uint8_t
#endif // __cplusplus
{
    /// Seconds.
    QkDelayUnit_S = 0,
    /// Milliseconds.
    QkDelayUnit_MS = 1,
    /// Microseconds.
    QkDelayUnit_US = 2,
    /// Nanoseconds.
    QkDelayUnit_NS = 3,
    /// Picoseconds.
    QkDelayUnit_PS = 4,
    /// Device-native time unit ``dt``.
    QkDelayUnit_DT = 5,
};
#ifndef __cplusplus
typedef uint8_t QkDelayUnit;
#endif // __cplusplus

enum QkGate
#ifdef __cplusplus
    : uint8_t
#endif // __cplusplus
{
    QkGate_GlobalPhase = 0,
    QkGate_H = 1,
    QkGate_I = 2,
    QkGate_X = 3,
    QkGate_Y = 4,
    QkGate_Z = 5,
    QkGate_Phase = 6,
    QkGate_R = 7,
    QkGate_RX = 8,
    QkGate_RY = 9,
    QkGate_RZ = 10,
    QkGate_S = 11,
    QkGate_Sdg = 12,
    QkGate_SX = 13,
    QkGate_SXdg = 14,
    QkGate_T = 15,
    QkGate_Tdg = 16,
    QkGate_U = 17,
    QkGate_U1 = 18,
    QkGate_U2 = 19,
    QkGate_U3 = 20,
    QkGate_CH = 21,
    QkGate_CX = 22,
    QkGate_CY = 23,
    QkGate_CZ = 24,
    QkGate_DCX = 25,
    QkGate_ECR = 26,
    QkGate_Swap = 27,
    QkGate_ISwap = 28,
    QkGate_CPhase = 29,
    QkGate_CRX = 30,
    QkGate_CRY = 31,
    QkGate_CRZ = 32,
    QkGate_CS = 33,
    QkGate_CSdg = 34,
    QkGate_CSX = 35,
    QkGate_CU = 36,
    QkGate_CU1 = 37,
    QkGate_CU3 = 38,
    QkGate_RXX = 39,
    QkGate_RYY = 40,
    QkGate_RZZ = 41,
    QkGate_RZX = 42,
    QkGate_XXMinusYY = 43,
    QkGate_XXPlusYY = 44,
    QkGate_CCX = 45,
    QkGate_CCZ = 46,
    QkGate_CSwap = 47,
    QkGate_RCCX = 48,
    QkGate_C3X = 49,
    QkGate_C3SX = 50,
    QkGate_RC3X = 51,
};
#ifndef __cplusplus
typedef uint8_t QkGate;
#endif // __cplusplus

/// A container for :class:`.QuantumCircuit` instruction listings that stores
/// :class:`.CircuitInstruction` instances in a packed form by interning
/// their :attr:`~.CircuitInstruction.qubits` and
/// :attr:`~.CircuitInstruction.clbits` to native vectors of indices.
///
/// Before adding a :class:`.CircuitInstruction` to this container, its
/// :class:`.Qubit` and :class:`.Clbit` instances MUST be registered via the
/// constructor or via :meth:`.CircuitData.add_qubit` and
/// :meth:`.CircuitData.add_clbit`. This is because the order in which
/// bits of the same type are added to the container determines their
/// associated indices used for storage and retrieval.
///
/// Once constructed, this container behaves like a Python list of
/// :class:`.CircuitInstruction` instances. However, these instances are
/// created and destroyed on the fly, and thus should be treated as ephemeral.
///
/// For example,
///
/// .. plot::
///    :include-source:
///    :no-figs:
///
///     qubits = [Qubit()]
///     data = CircuitData(qubits)
///     data.append(CircuitInstruction(XGate(), (qubits[0],), ()))
///     assert(data[0] == data[0]) # => Ok.
///     assert(data[0] is data[0]) # => PANICS!
///
/// .. warning::
///
///     This is an internal interface and no part of it should be relied upon
///     outside of Qiskit.
///
/// Args:
///     qubits (Iterable[:class:`.Qubit`] | None): The initial sequence of
///         qubits, used to map :class:`.Qubit` instances to and from its
///         indices.
///     clbits (Iterable[:class:`.Clbit`] | None): The initial sequence of
///         clbits, used to map :class:`.Clbit` instances to and from its
///         indices.
///     data (Iterable[:class:`.CircuitInstruction`]): An initial instruction
///         listing to add to this container. All bits appearing in the
///         instructions in this iterable must also exist in ``qubits`` and
///         ``clbits``.
///     reserve (int): The container's initial capacity. This is reserved
///         before copying instructions into the container when ``data``
///         is provided, so the initialized container's unused capacity will
///         be ``max(0, reserve - len(data))``.
///
/// Raises:
///     KeyError: if ``data`` contains a reference to a bit that is not present
///         in ``qubits`` or ``clbits``.
typedef struct QkCircuit QkCircuit;

/// An individual operation count represented by the operation name
/// and the number of instances in the circuit.
typedef struct {
    /// A nul terminated string representing the operation name
    const char *name;
    /// The number of instances of this operation in the circuit
    uintptr_t count;
} QkOpCount;

/// An array of ``OpCount`` objects representing the total counts of all
/// the operation types in a circuit.
typedef struct {
    /// A array of size ``len`` containing ``OpCount`` objects for each
    /// type of operation in the circuit
    QkOpCount *data;
    /// The number of elements in ``data``
    uintptr_t len;
} QkOpCounts;

/// A circuit instruction representation.
///
/// This struct represents the data contained in an individual instruction in a ``QkCircuit``.
/// It is not a pointer to the underlying object, but contains a copy of the properties of the
/// instruction for inspection.
typedef struct {
    /// The instruction name
    const char *name;
    /// The number of qubits for this instruction.
    uint32_t num_qubits;
    /// A pointer to an array of qubit indices this instruction operates on.
    uint32_t *qubits;
    /// The number of clbits for this instruction.
    uint32_t num_clbits;
    /// A pointer to an array of clbit indices this instruction operates on.
    uint32_t *clbits;
    /// The number of parameters for this instruction.
    uint32_t num_params;
    /// A pointer to an array of parameter values for this instruction.
    double *params;
} QkCircuitInstruction;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// @ingroup QkCircuit
/// Construct a new circuit with the given number of qubits and clbits.
///
/// @param num_qubits The number of qubits the circuit contains.
/// @param num_clbits The number of clbits the circuit contains.
///
/// @return A pointer to the created circuit.
///
/// # Example
///
///     QkCircuit *empty = qk_circuit_new(100, 100);
///
QkCircuit *qk_circuit_new(uint32_t num_qubits, uint32_t num_clbits);

/// @ingroup QkCircuit
/// Get the number of qubits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t num_qubits = qk_circuit_num_qubits(qc);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
uint32_t qk_circuit_num_qubits(const QkCircuit *circuit);

/// @ingroup QkCircuit
/// Get the number of clbits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 50);
///     uint32_t num_clbits = qk_circuit_num_clbits(qc);  // num_clbits==50
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
uint32_t qk_circuit_num_clbits(const QkCircuit *circuit);

/// @ingroup QkCircuit
/// Free the circuit.
///
/// @param circuit A pointer to the circuit to free.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     qk_circuit_free(qc);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not either null or a valid pointer to a
/// ``QkCircuit``.
void qk_circuit_free(QkCircuit *circuit);

/// @ingroup QkCircuit
/// Append a standard gate to the circuit.
///
/// @param circuit A pointer to the circuit to add the gate to.
/// @param gate The StandardGate to add to the circuit.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``double`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check ``qk_gate_num_qubits`` and ``qk_gate_num_params`` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkExitCode qk_circuit_gate(QkCircuit *circuit, QkGate gate, const uint32_t *qubits,
                           const double *params);

/// @ingroup QkCircuit
/// Get the number of qubits for a ``QkGate``.
///
/// @param gate The standard gate to get the number of qubits for.
///
/// @return The number of qubits the gate acts on.
///
/// # Example
///
///     uint32_t num_qubits = qk_gate_num_qubits(QkGate_CCX);
///
uint32_t qk_gate_num_qubits(QkGate gate);

/// @ingroup QkCircuit
/// Get the number of parameters for a ``QkGate``.
///
/// @param gate The standard gate to get the number of qubits for.
///
/// @return The number of parameters the gate has.
///
/// # Example
///
///     uint32_t num_params = qk_gate_num_params(QkGate_R);
///
uint32_t qk_gate_num_params(QkGate gate);

/// @ingroup QkCircuit
/// Append a measurement to the circuit
///
/// @param circuit A pointer to the circuit to add the measurement to
/// @param qubit The ``uint32_t`` for the qubit to measure
/// @param clbit The ``uint32_t`` for the clbit to store the measurement outcome in
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     qk_circuit_measure(qc, 0, 0);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkExitCode qk_circuit_measure(QkCircuit *circuit, uint32_t qubit, uint32_t clbit);

/// @ingroup QkCircuit
/// Append a reset to the circuit
///
/// @param circuit A pointer to the circuit to add the reset to
/// @param qubit The ``uint32_t`` for the qubit to reset
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     qk_circuit_reset(qc, 0);
///
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkExitCode qk_circuit_reset(QkCircuit *circuit, uint32_t qubit);

/// @ingroup QkCircuit
/// Append a barrier to the circuit.
///
/// @param circuit A pointer to the circuit to add the barrier to.
/// @param num_qubits The number of qubits wide the barrier is.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the barrier on.
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubits[5] = {0, 1, 2, 3, 4};
///     qk_circuit_barrier(qc, qubits, 5);
///
/// # Safety
///
/// The length of the array ``qubits`` points to must be ``num_qubits``. If there is
/// a mismatch the behavior is undefined.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkExitCode qk_circuit_barrier(QkCircuit *circuit, const uint32_t *qubits, uint32_t num_qubits);

/// @ingroup QkCircuit
/// Return a list of string names for instructions in a circuit and their counts.
///
/// @param circuit A pointer to the circuit to get the counts for.
///
/// @return An ``OpCounts`` struct containing the circuit operation counts.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubits, NULL);
///     QkOpCounts counts = qk_circuit_count_ops(qc);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkOpCounts qk_circuit_count_ops(const QkCircuit *circuit);

/// @ingroup QkCircuit
/// Return the total number of instructions in the circuit.
///
/// @param circuit A pointer to the circuit to get the total number of instructions for.
///
/// @return The total number of instructions in the circuit.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     size_t num = qk_circuit_num_instructions(qc); // 1
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
uintptr_t qk_circuit_num_instructions(const QkCircuit *circuit);

/// @ingroup QkCircuit
/// Return the instruction details for an instruction in the circuit.
///
/// This function is used to get the instruction details for a given instruction in
/// the circuit.
///
/// @param circuit A pointer to the circuit to get the instruction details for.
/// @param index The instruction index to get the instruction details of.
///
/// @return The instruction details for the specified instructions.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     QkCircuitInstruction inst = qk_circuit_get_instruction(qc, 0);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by ``qk_circuit_num_instructions``
/// otherwise this function will panic.
QkCircuitInstruction qk_circuit_get_instruction(const QkCircuit *circuit, uintptr_t index);

/// @ingroup QkCircuit
/// Free a circuit instruction object.
///
/// @param inst The instruction to free.
///
/// # Safety
///
/// Behavior is undefined if ``inst`` is not an object returned by ``qk_circuit_get_instruction``.
void qk_circuit_instruction_free(QkCircuitInstruction inst);

/// @ingroup QkCircuit
/// Free a circuit op count list.
///
/// @param op_counts The returned op count list from ``qk_circuit_count_ops``.
///
/// # Safety
///
/// Behavior is undefined if ``op_counts`` is not the object returned by ``qk_circuit_count_ops``.
void qk_opcounts_free(QkOpCounts op_counts);

#ifdef QISKIT_C_PYTHON_INTERFACE
/// @ingroup QkCircuit
/// Convert to a Python-space ``QuantumCircuit``.
///
/// This function takes ownership of the pointer and gives it to Python. Using
/// the input ``circuit`` pointer after it's passed to this function is
/// undefined behavior. In particular, ``qk_circuit_free`` should not be called
/// on this pointer anymore.
///
/// @param circuit The C-space ``QkCircuit`` pointer.
///
/// @return A Python ``QuantumCircuit`` object.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to
/// a ``QkCircuit``
///
/// It is assumed that the thread currently executing this function holds the
/// Python GIL. This is required to create the Python object returned by this
/// function.
PyObject *qk_circuit_to_python(QkCircuit *circuit);
#endif

/// @ingroup QkCircuit
/// Append a delay instruction to the circuit.
///
/// @param circuit A pointer to the circuit to add the delay to.
/// @param qubit The ``uint32_t`` index of the qubit to apply the delay to.
/// @param duration The duration of the delay.
/// @param unit An enum representing the unit of the duration.
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///     qk_circuit_delay(qc, 0, 100.0, QkDelayUnit_NS);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
QkExitCode qk_circuit_delay(QkCircuit *circuit, uint32_t qubit, double duration, QkDelayUnit unit);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // QISKIT__CIRCUIT_H
