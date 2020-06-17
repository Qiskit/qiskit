# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Methods to assist with compiling tasks.
    These functions are all used by a_star_mapper (Python, not Cython) in order to such things
    as KAK refactoring, flip swaps to satisfy the coupling map, evaluate the cost of a circuit, etc.
"""

# import math
import numpy as np

from qiskit.providers.builtinsimulators._simulatortools import single_gate_matrix
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.mapper._compiling import euler_angles_1q, simplify_U, two_qubit_kak
from qiskit.mapper._mapping import MapperError


def cost_of_group(gates, gate_costs):
    """determine the cost of all gates in the group"""
    cost = 0
    for gate in gates:
        cost += gate_costs[gate["name"]]
    return cost


def count_cx_gates(gates):
    """count the number of cx gates"""
    count = 0
    for gate in gates:
        if gate["name"].lower() == "cx":
            count += 1
    return count


def apply_gates(compiled_dag, gates, qubit0, qubit1, coupling_map):
    """optimize gates and add them to the compiled circuit"""

    from sympy import sympify

    # fix the direction of the CX gates
    gates_fixed = []
    for gate in gates:
        if len(gate["qargs"]) == 2:
            if (gate["qargs"][0][1],
                gate["qargs"][1][1]) == (
                        qubit1, qubit0) and (("q", qubit1),
                                             ("q", qubit0)) not in coupling_map:

                # swap the direction of the CNOT gate to satisfy constraints
                # given by the coupling map
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", qubit0)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", qubit1)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                gate["qargs"] = [gate["qargs"][1], gate["qargs"][0]]
                gates_fixed += [gate]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", qubit0)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", qubit1)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
            elif (gate["qargs"][0][1], gate["qargs"][1][1]) == (qubit0, qubit1):
                gates_fixed += [gate]
        else:
            gates_fixed += [gate]

    # combine consecutive single qubit gates and simplify the result
    u0_mat = np.identity(2)
    u0_gates = []
    u1_mat = np.identity(2)
    u1_gates = []
    for gate in gates_fixed:
        if gate["name"] == "cx":
            # add single qubit gates to the compiled circuit
            if not np.array_equal(u0_mat, np.identity(2)):
                try:
                    theta, phi, lamb = OneQubitEulerDecomposer(u0_mat).angles()
                    name, params, _ = simplify_U(theta, phi, lamb)
                    if name == "u3":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", qubit0)],
                            [],
                            [
                                sympify(params[0]),
                                sympify(params[1]),
                                sympify(params[2]),
                            ],
                            None,
                        )
                    elif name == "u2":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", qubit0)],
                            [],
                            [sympify(params[1]), sympify(params[2])],
                            None,
                        )
                    elif name == "u1":
                        compiled_dag.apply_operation_back(
                            name, [("q", qubit0)], [], [sympify(params[2])], None
                        )
                except MapperError as _:
                    # fallback if decomposition provided by qiskit fails
                    for inner_gate in u0_gates:
                        compiled_dag.apply_operation_back(
                            inner_gate["name"],
                            inner_gate["qargs"],
                            inner_gate["cargs"],
                            inner_gate["params"],
                            inner_gate["condition"],
                        )
                u0_mat = np.identity(2)
                u0_gates = []
            if not np.array_equal(u1_mat, np.identity(2)):
                try:
                    theta, phi, lamb, _ = euler_angles_1q(u1_mat)
                    name, params, _ = simplify_U(theta, phi, lamb)
                    if name == "u3":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", qubit1)],
                            [],
                            [
                                sympify(params[0]),
                                sympify(params[1]),
                                sympify(params[2]),
                            ],
                            None,
                        )
                    elif name == "u2":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", qubit1)],
                            [],
                            [sympify(params[1]), sympify(params[2])],
                            None,
                        )
                    elif name == "u1":
                        compiled_dag.apply_operation_back(
                            name, [("q", qubit1)], [], [sympify(params[2])], None
                        )
                except MapperError as _:
                    # fallback if decomposition provided by qiskit fails
                    for inner_gate in u1_gates:
                        compiled_dag.apply_operation_back(
                            inner_gate["name"],
                            inner_gate["qargs"],
                            inner_gate["cargs"],
                            inner_gate["params"],
                            inner_gate["condition"],
                        )
                u1_mat = np.identity(2)
                u1_gates = []
            compiled_dag.apply_operation_back(
                gate["name"],
                gate["qargs"],
                gate["cargs"],
                gate["params"],
                gate["condition"],
            )

        else:
            # combine single gates
            if gate["qargs"][0][1] == qubit0:
                u0_mat = np.dot(
                    single_gate_matrix(gate["name"], gate["params"]), u0_mat
                )
                u0_gates += [gate]
            else:
                u1_mat = np.dot(
                    single_gate_matrix(gate["name"], gate["params"]), u1_mat
                )
                u1_gates += [gate]

    # add single qubit gates to the compiled circuit
    if not np.array_equal(u0_mat, np.identity(2)):
        try:
            theta, phi, lamb, _ = euler_angles_1q(u0_mat)
            name, params, _ = simplify_U(theta, phi, lamb)
            if name == "u3":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", qubit0)],
                    [],
                    [sympify(params[0]), sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u2":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", qubit0)],
                    [],
                    [sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u1":
                compiled_dag.apply_operation_back(
                    name, [("q", qubit0)], [], [sympify(params[2])], None
                )
        except MapperError as _:
            # fallback if decomposition provided by qiskit fails
            for inner_gate in u0_gates:
                compiled_dag.apply_operation_back(
                    inner_gate["name"],
                    inner_gate["qargs"],
                    inner_gate["cargs"],
                    inner_gate["params"],
                    inner_gate["condition"],
                )
        u0_mat = np.identity(2)
        u0_gates = []
    if not np.array_equal(u1_mat, np.identity(2)):
        try:
            theta, phi, lamb, _ = euler_angles_1q(u1_mat)
            name, params, _ = simplify_U(theta, phi, lamb)
            if name == "u3":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", qubit1)],
                    [],
                    [sympify(params[0]), sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u2":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", qubit1)],
                    [],
                    [sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u1":
                compiled_dag.apply_operation_back(
                    name, [("q", qubit1)], [], [sympify(params[2])], None
                )
        except MapperError as _:
            # fallback if decomposition provided by qiskit fails
            for inner_gate in u1_gates:
                compiled_dag.apply_operation_back(
                    inner_gate["name"],
                    inner_gate["qargs"],
                    inner_gate["cargs"],
                    inner_gate["params"],
                    inner_gate["condition"],
                )
        u1_mat = np.identity(2)
        u1_gates = []
    # return compiled circuit
    return compiled_dag


def optimize_gate_groups(grouped_gates, coupling_map, empty_dag, gate_costs):
    """ Optimize groups of gates and add them to the compiled circuit """

    from sympy import sympify

    compiled_dag = empty_dag

    import networkx as nx

    gate_groups = nx.topological_sort(grouped_gates)

    # iterate over all gates in the group and build corresponding 4x4 unitary matrix
    for index in gate_groups:
        group = grouped_gates.node[index]
        if len(group["qubits"]) == 2:
            cost_before = cost_of_group(group["gates"], gate_costs)
            qubits = [group["qubits"][0], group["qubits"][1]]
            if (("q", qubits[0]), ("q", qubits[1])) in coupling_map:
                qubits = [group["qubits"][0], group["qubits"][1]]
            else:
                if (("q", qubits[1]), ("q", qubits[0])) in coupling_map:
                    qubits = [group["qubits"][1], group["qubits"][0]]

            # estimate whether KAK decomposition (which is time intensive) will improve the result
            if count_cx_gates(group["gates"]) <= 3:
                compiled_dag = apply_gates(
                    compiled_dag, group["gates"], qubits[0], qubits[1], coupling_map
                )
                continue

            u_mat = np.identity(4)
            cnot1 = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
            )
            cnot2 = np.array(
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
            )
            for gate in group["gates"]:
                if gate["name"] == "cx":
                    if gate["qargs"][0][1] == qubits[0]:
                        u_mat = np.dot(cnot1, u_mat)
                    else:
                        u_mat = np.dot(cnot2, u_mat)
                else:
                    if gate["qargs"][0][1] == qubits[0]:
                        u_mat = np.dot(
                            np.kron(
                                single_gate_matrix(gate["name"], gate["params"]),
                                np.identity(2),
                            ),
                            u_mat,
                        )
                    else:
                        u_mat = np.dot(
                            np.kron(
                                np.identity(2),
                                single_gate_matrix(gate["name"], gate["params"]),
                            ),
                            u_mat,
                        )

            new_gates = []
            try:
                for gate in two_qubit_kak(u_mat):
                    i_0 = gate["args"][0]
                    if gate["name"] == "cx":
                        i_1 = gate["args"][1]
                        new_gates += [
                            {
                                "type": "op",
                                "name": "cx",
                                "qargs": [("q", qubits[i_0]), ("q", qubits[i_1])],
                                "cargs": [],
                                "params": [],
                                "condition": None,
                            }
                        ]
                    elif gate["name"] == "u1":
                        new_gates += [
                            {
                                "type": "op",
                                "name": "u1",
                                "qargs": [("q", qubits[i_0])],
                                "cargs": [],
                                "params": [sympify(gate["params"][2])],
                                "condition": None,
                            }
                        ]
                    elif gate["name"] == "u2":
                        new_gates += [
                            {
                                "type": "op",
                                "name": "u2",
                                "qargs": [("q", qubits[i_0])],
                                "cargs": [],
                                "params": [
                                    sympify(gate["params"][1]),
                                    sympify(gate["params"][2]),
                                ],
                                "condition": None,
                            }
                        ]
                    elif gate["name"] == "u3":
                        new_gates += [
                            {
                                "type": "op",
                                "name": "u3",
                                "qargs": [("q", qubits[i_0])],
                                "cargs": [],
                                "params": [
                                    sympify(gate["params"][0]),
                                    sympify(gate["params"][1]),
                                    sympify(gate["params"][2]),
                                ],
                                "condition": None,
                            }
                        ]

                # determine cost
                cost_after = cost_of_group(new_gates, gate_costs)
            except (MapperError, ValueError) as _:
                cost_after = cost_before + 1
            if cost_after < cost_before:
                # add gates in the decomposition
                compiled_dag = apply_gates(
                    compiled_dag, new_gates, qubits[0], qubits[1], coupling_map
                )
            else:
                # add original gates
                compiled_dag = apply_gates(
                    compiled_dag, group["gates"], qubits[0], qubits[1], coupling_map
                )
        else:
            # apply measurement gates and barrier gates without optimization
            for gate in group["gates"]:
                compiled_dag.apply_operation_back(
                    gate["name"],
                    gate["qargs"],
                    gate["cargs"],
                    gate["params"],
                    gate["condition"],
                )

    return compiled_dag
