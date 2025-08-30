# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""FFI to call qiskit C API."""

import ctypes

import numpy

import qiskit

LIB_PATH = qiskit._accelerate.__file__
LIB = ctypes.PyDLL(LIB_PATH)


class QkCircuit(ctypes.Structure):
    """QkCircuit Opaque Type"""

    pass


class QkTarget(ctypes.Structure):
    """QkTarget Opaque Type"""

    pass


class QkTargetEntry(ctypes.Structure):
    """QkTargetEntry Opaque Type"""

    pass


class QkTranspileLayout(ctypes.Structure):
    """QkTranspileLayout Opaque Type"""

    pass


class QkTranspileOptions(ctypes.Structure):
    """QkTranspileOptions struct"""

    _fields_ = [
        ("optimization_level", ctypes.c_uint8),
        ("seed", ctypes.c_int64),
        ("approximation_degree", ctypes.c_double),
    ]


class QkTranspileResult(ctypes.Structure):
    """QkTranspileResult struct"""

    _fields_ = [
        ("circuit", ctypes.POINTER(QkCircuit)),
        ("layout", ctypes.POINTER(QkTranspileLayout)),
    ]


LIB.qk_circuit_new.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
LIB.qk_circuit_new.restype = ctypes.POINTER(QkCircuit)
LIB.qk_circuit_gate.argtypes = [
    ctypes.POINTER(QkCircuit),
    ctypes.c_uint8,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_double),
]
LIB.qk_circuit_measure.argtypes = [ctypes.POINTER(QkCircuit), ctypes.c_uint32, ctypes.c_uint32]
LIB.qk_circuit_to_python.argtypes = [
    ctypes.POINTER(QkCircuit),
]
LIB.qk_circuit_barrier.argtypes = [
    ctypes.POINTER(QkCircuit),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_uint32,
]

LIB.qk_circuit_to_python.restype = ctypes.py_object
LIB.qk_target_new.argtypes = [
    ctypes.c_uint32,
]
LIB.qk_target_new.restype = ctypes.POINTER(QkTarget)
LIB.qk_target_entry_new.argtypes = [
    ctypes.c_uint8,
]
LIB.qk_target_entry_new.restype = ctypes.POINTER(QkTargetEntry)
LIB.qk_target_entry_new_measure.argtypes = []
LIB.qk_target_entry_new_measure.restype = ctypes.POINTER(QkTargetEntry)
LIB.qk_target_entry_add_property.argtypes = [
    ctypes.POINTER(QkTargetEntry),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_uint32,
    ctypes.c_double,
    ctypes.c_double,
]
LIB.qk_target_add_instruction.argtypes = [ctypes.POINTER(QkTarget), ctypes.POINTER(QkTargetEntry)]
LIB.qk_transpile.argtypes = [
    ctypes.POINTER(QkCircuit),
    ctypes.POINTER(QkTarget),
    ctypes.POINTER(QkTranspileOptions),
    ctypes.POINTER(QkTranspileResult),
    ctypes.POINTER(ctypes.c_char_p),
]
LIB.qk_transpile.restype = ctypes.c_uint32
LIB.qk_transpile_layout_to_python.argtypes = [
    ctypes.POINTER(QkTranspileLayout),
    ctypes.POINTER(QkCircuit),
]
LIB.qk_transpile_layout_to_python.restype = ctypes.py_object


def build_circuit_from_python(circuit: qiskit.circuit.QuantumCircuit) -> ctypes.POINTER(QkCircuit):
    """Convert a Python circuit to a C circuit if compatible."""

    c_qc = LIB.qk_circuit_new(circuit.num_qubits, circuit.num_clbits)
    for inst in circuit.data:
        qubit_idx = [circuit.find_bit(x).index for x in inst.qubits]
        qubits = (ctypes.c_uint32 * len(qubit_idx))(*qubit_idx)
        params = (ctypes.c_double * len(inst.params))(*inst.params)
        if isinstance(inst.operation, qiskit.circuit.Measure):
            clbit = circuit.find_bit(inst.clbits[0]).index
            LIB.qk_circuit_measure(c_qc, qubit_idx[0], clbit)
        elif isinstance(inst.operation, qiskit.circuit.Barrier):
            LIB.qk_circuit_barrier(
                c_qc, ctypes.cast(qubits, ctypes.POINTER(ctypes.c_uint32)), len(qubit_idx)
            )
        else:
            LIB.qk_circuit_gate(
                c_qc,
                int(inst.operation._standard_gate),
                ctypes.cast(qubits, ctypes.POINTER(ctypes.c_uint32)),
                ctypes.cast(params, ctypes.POINTER(ctypes.c_double)),
            )
    return c_qc


def build_homogenous_target(
    cmap: qiskit.transpiler.CouplingMap, basis_gates: list[str], seed: int, ideal_gates=False
) -> QkTarget:
    """Build a target with a homogenous gate set."""

    name_mapping = qiskit.circuit.library.standard_gates.get_standard_gate_name_mapping()
    c_target = LIB.qk_target_new(cmap.size())
    rng = numpy.random.default_rng(seed)
    for gate in basis_gates:
        gate_obj = name_mapping[gate]
        entry = LIB.qk_target_entry_new(int(gate_obj._standard_gate))
        if gate_obj.num_qubits == 2:
            for edge in cmap.get_edges():
                qubits = (ctypes.c_uint32 * 2)(*edge)
                if not ideal_gates:
                    error = rng.uniform(0.0, 1.0)
                    duration = rng.uniform(0.0, 1.0)
                else:
                    error = float("nan")
                    duration = float("nan")
                LIB.qk_target_entry_add_property(
                    entry, ctypes.cast(qubits, ctypes.POINTER(ctypes.c_uint32)), 2, duration, error
                )
        elif gate_obj.num_qubits == 1:
            for qubit in range(cmap.size()):
                qubits = (ctypes.c_int32 * 1)(qubit)
                if not ideal_gates:
                    error = rng.uniform(0.0, 1.0)
                    duration = rng.uniform(0.0, 1.0)
                else:
                    error = float("nan")
                    duration = float("nan")
                LIB.qk_target_entry_add_property(
                    entry, ctypes.cast(qubits, ctypes.POINTER(ctypes.c_uint32)), 1, duration, error
                )
        else:
            raise qiskit.transpiler.exceptions.TranspilerError(f"Invalid gate: {gate}")
        LIB.qk_target_add_instruction(c_target, entry)
    measure_entry = LIB.qk_target_entry_new_measure()
    for qubit in range(cmap.size()):
        qubits = (ctypes.c_int32 * 1)(qubit)
        if not ideal_gates:
            error = rng.uniform(0.0, 1.0)
            duration = rng.uniform(0.0, 1.0)
        else:
            error = float("nan")
            duration = float("nan")

        LIB.qk_target_entry_add_property(
            measure_entry, ctypes.cast(qubits, ctypes.POINTER(ctypes.c_uint32)), 1, duration, error
        )
    LIB.qk_target_add_instruction(c_target, measure_entry)
    return c_target


def transpile_from_c(
    circuit: ctypes.POINTER(QkCircuit),
    target: ctypes.POINTER(QkTarget),
    optimization_level,
    approximation_degree,
    seed,
) -> qiskit.QuantumCircuit:
    """Transpile a circuit and return it as a python circuit."""
    args = []
    if optimization_level is not None:
        args.append(optimization_level)
    else:
        args.append(2)
    if seed is not None:
        args.append(seed)
    else:
        args.append(-1)
    if approximation_degree is not None:
        args.append(approximation_degree)
    else:
        args.append(1.0)
    options = QkTranspileOptions(*args)
    result = QkTranspileResult(ctypes.POINTER(QkCircuit)(), ctypes.POINTER(QkTranspileLayout)())
    error = ctypes.pointer(ctypes.c_char_p(None))
    res = LIB.qk_transpile(
        circuit,
        target,
        ctypes.pointer(options),
        ctypes.pointer(result),
        error,
    )
    if res != 0:
        raise qiskit.transpiler.exceptions.TranspilerError(
            f"Transpilation failed: {error.contents.value.decode('utf8')}"
        )
    layout = LIB.qk_transpile_layout_to_python(result.layout, result.circuit)
    out = LIB.qk_circuit_to_python(result.circuit)
    out._layout = layout
    return out
