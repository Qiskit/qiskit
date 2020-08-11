# -*- coding: utf-8 -*-

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


"""Helper function for converting a circuit to a gate"""
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.exceptions import QiskitError


def circuit_to_gate(circuit, parameter_map=None, equivalence_library=None, label=None):
    """Build a ``Gate`` object from a ``QuantumCircuit``.

    The gate is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The gate will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the gate.
           If None, existing circuit parameters will also parameterize the
           Gate.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted gate will be registered.
        label (str): Optional gate label.

    Raises:
        QiskitError: if circuit is non-unitary or if
            parameter_map is not compatible with circuit

    Return:
        Gate: a Gate equivalent to the action of the
        input circuit. Upon decomposition, this gate will
        yield the components comprising the original circuit.
    """
    # pylint: disable=cyclic-import
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    if circuit.clbits:
        raise QiskitError('Circuit with classical bits cannot be converted '
                          'to gate.')

    for inst, _, _ in circuit.data:
        if not isinstance(inst, Gate):
            raise QiskitError(('One or more instructions cannot be converted to'
                               ' a gate. "{}" is not a gate instruction').format(
                                   inst.name))

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(('parameter_map should map all circuit parameters. '
                           'Circuit parameters: {}, parameter_map: {}').format(
                               circuit.parameters, parameter_dict))

    gate = Gate(name=circuit.name,
                num_qubits=sum([qreg.size for qreg in circuit.qregs]),
                params=sorted(parameter_dict.values(), key=lambda p: p.name),
                label=label)
    gate.condition = None

    def find_bit_position(bit):
        """find the index of a given bit (Register, int) within
        a flat ordered list of bits of the circuit
        """
        if isinstance(bit, Qubit):
            ordered_regs = circuit.qregs
        else:
            ordered_regs = circuit.cregs
        reg_index = ordered_regs.index(bit.register)
        return sum([reg.size for reg in ordered_regs[:reg_index]]) + bit.index

    target = circuit.assign_parameters(parameter_dict, inplace=False)

    if equivalence_library is not None:
        equivalence_library.add_equivalence(gate, target)

    rules = target.data

    if gate.num_qubits > 0:
        q = QuantumRegister(gate.num_qubits, 'q')

    # The 3rd parameter in the output tuple) is hard coded to [] because
    # Gate objects do not have cregs set and we've verified that all
    # instructions are gates
    rules = list(map(
        lambda x: (x[0],
                   list(map(lambda y: q[find_bit_position(y)], x[1])),
                   []),
        rules))
    qc = QuantumCircuit(q, name=gate.name, global_phase=target.global_phase)
    qc._data = rules
    gate.definition = qc
    return gate
