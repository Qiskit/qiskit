# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=============================================
Quickly simulating circuits (:mod:`qiskit.providers.quick_simulate`)
=============================================
.. currentmodule:: qiskit.providers.quick_simulate
.. autofunction:: quick_simulate
"""


from qiskit.providers.basicaer import BasicAer
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.execute import execute
from qiskit.exceptions import QiskitError


def quick_simulate(circuit, shots=1024, x="0", verbose=False):
    """simulates circuit with given input

    Args:
        circuit (QuantumCircuit): Circuit to simulate. Currently no more than 2 registers supported
        shots (int, optional): number of shots to simulate. Default 1024
        x (str, optional): input string eg "11" would apply an X gate to first 2 qubits. Default "0"
        verbose (bool, optional): prints extra output

    Returns:
        dict[str:int]: results.get_counts()

    Raises:
        QiskitError: if circuit has more than two registers
    """
    names = []
    regs = []
    for q in circuit.qubits:
        name = q.register.name
        size = len(q.register)
        if name not in names:
            names.append(name)
            regs.append(size)

    if verbose:
        print(names, regs)

    # assuming that we only have 2: control + ancillary

    qra = QuantumRegister(regs[0], name=names[0])
    if len(regs) == 1:
        qa = QuantumCircuit(qra)
    elif len(regs) == 2:
        qran = QuantumRegister(regs[1], name=names[1])
        qa = QuantumCircuit(qra, qran)
    else:
        raise QiskitError("Not yet implemented for more than 2 registers")

    if len(x) != sum(regs):
        x += "0" * (sum(regs) - len(x))
    if verbose:
        print(x)
    for i, bit in enumerate(x):
        if verbose:
            print(bit, type(bit))
        if bit != "0":
            qa.x(i)
    qa.barrier()

    qa.extend(circuit)

    if verbose:
        print(qa)

    backend = BasicAer.get_backend('qasm_simulator')
    results = execute(qa, backend=backend, shots=shots).result()
    answer = results.get_counts()
    return answer
