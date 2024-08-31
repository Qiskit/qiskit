# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for generating random circuits."""

import numpy as np

from qiskit.circuit import  QuantumCircuit
from qiskit.circuit.library import standard_gates
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVectorElement


def random_circuit(
    num_qubits: int, 
    num_gates: int, 
    gates: str|list[str] ="all", 
    parameterized: bool = False, 
    measure: bool = False, 
    seed: int|np.random.Generator = None
):
    """
    Generate a pseudo-random quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        num_gates (int): The number of gates in the circuit.
        gates (list[str]): The gates to use in the circuit.
            If ``"all"``, use all the gates in the standard library.
            If ``"Clifford"``, use the gates in the Clifford set.
            If ``"Clifford+T"``, use the gates in the Clifford set and the T gate.
        parameterized (bool): Whether to parameterize the gates. Defaults to False.
        measure (bool): Whether to add a measurement at the end of the circuit. Defaults to False.
        seed (int or numpy.random.Generator): The seed for the random number generator. Defaults to None.
    """
    instructions = standard_gates.get_standard_gate_name_mapping()
    
    if gates == "all":
        gates = list(instructions.keys())
        gates.remove('delay')
        if measure is False:
            gates.remove('measure')
    elif gates == "Clifford":
        gates = ["x","y","z","h","s","sdg","sx","sxdg","cx","cy","cz","ch"]
    elif gates == "Clifford+T":
        gates = ["x","y","z","h","s","sdg","t","tdg","sx","sxdg","cx","cy","cz","ch"]
        
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits) if measure is False else QuantumCircuit(num_qubits, num_qubits)

    param_base = Parameter('ϴ')
    num_params = 0

    for name in samples:
        gate, nqargs = instructions[name], instructions[name].num_qubits

        if (len_param:=len(gate.params)) >0:
            gate = gate.copy()
            if parameterized is True:
                gate.params = [ParameterVectorElement(param_base, num_params + i) for i in range(len_param)]
                num_params += len_param
            else:
                param_list = rng.choice(range(1,16), 2*len_param)
                gate.params = [param_list[2*i]/param_list[2*i+1]*np.pi for i in range(len_param)]

        qargs = rng.choice(range(num_qubits), nqargs, replace = False).tolist()
        print(gate, qargs)
        circ.append(gate, qargs, copy=False)

    return circ
