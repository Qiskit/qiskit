# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a circuit to an instruction."""
from qiskit.circuit.parametertable import ParameterTable, ParameterReferences
from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit


def circuit_to_instruction(circuit, parameter_map=None, equivalence_library=None, label=None):
    """Build an :class:`~.circuit.Instruction` object from a :class:`.QuantumCircuit`.

    The instruction is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The instruction will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the instruction.
           If None, existing circuit parameters will also parameterize the
           instruction.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted instruction will be registered.
        label (str): Optional instruction label.

    Raises:
        QiskitError: if parameter_map is not compatible with circuit

    Return:
        qiskit.circuit.Instruction: an instruction equivalent to the action of the
        input circuit. Upon decomposition, this instruction will
        yield the components comprising the original circuit.

    Example:
        .. code-block::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.converters import circuit_to_instruction

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            circuit_to_instruction(circ)
    """
    # pylint: disable=cyclic-import
    from qiskit.circuit.quantumcircuit import QuantumCircuit

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(
            (
                "parameter_map should map all circuit parameters. "
                "Circuit parameters: {}, parameter_map: {}"
            ).format(circuit.parameters, parameter_dict)
        )

    out_instruction = Instruction(
        name=circuit.name,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        params=[*parameter_dict.values()],
        label=label,
    )
    out_instruction._condition = None

    target = circuit.assign_parameters(parameter_dict, inplace=False)

    if equivalence_library is not None:
        equivalence_library.add_equivalence(out_instruction, target)

    regs = []
    qreg, creg = None, None
    if out_instruction.num_qubits > 0:
        qreg = QuantumRegister(out_instruction.num_qubits, "q")
        regs.append(qreg)

    if out_instruction.num_clbits > 0:
        creg = ClassicalRegister(out_instruction.num_clbits, "c")
        regs.append(creg)

    clbit_map = {bit: creg[idx] for idx, bit in enumerate(circuit.clbits)}
    operation_map = {id(ParameterTable.GLOBAL_PHASE): ParameterTable.GLOBAL_PHASE}

    def fix_condition(op):
        original_id = id(op)
        if (out := operation_map.get(original_id)) is not None:
            return out

        condition = getattr(op, "condition", None)
        if condition:
            reg, val = condition
            if isinstance(reg, Clbit):
                op = op.c_if(clbit_map[reg], val)
            elif reg.size == creg.size:
                op = op.c_if(creg, val)
            else:
                raise QiskitError(
                    "Cannot convert condition in circuit with "
                    "multiple classical registers to instruction"
                )
        operation_map[original_id] = op
        return op

    data = target._data.copy()
    data.replace_bits(qubits=qreg, clbits=creg)
    data.map_ops(fix_condition)

    qc = QuantumCircuit(*regs, name=out_instruction.name)
    qc._data = data
    qc._parameter_table = ParameterTable(
        {
            param: ParameterReferences(
                (operation_map[id(operation)], param_index)
                for operation, param_index in target._parameter_table[param]
            )
            for param in target._parameter_table
        }
    )

    if circuit.global_phase:
        qc.global_phase = circuit.global_phase

    out_instruction.definition = qc

    return out_instruction
