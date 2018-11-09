"""
Methods to assist with compiling tasks.
"""
import math
import numpy as np

from qiskit.backends.aer._simulatortools import single_gate_matrix
from qiskit.mapper._compiling import euler_angles_1q, simplify_U, two_qubit_kak

# determine the cost of all gates in the group


def cost_of_group(gates, gate_costs):
    cost = 0
    for g in gates:
        cost += gate_costs[g["name"]]
    return cost


# count the number of cx gates
def count_cx_gates(gates):
    count = 0
    for g in gates:
        if g["name"].lower() == "cx":
            count += 1
    return count


# optimize gates and add them to the compiled circuit
def apply_gates(compiled_dag, gates, q0, q1):

    from sympy import sympify

    # fix the direction of the CX gates
    gates_fixed = []
    for g in gates:
        if len(g["qargs"]) == 2:
            if (g["qargs"][0][1], g["qargs"][1][1]) == (q1, q0):
                # swap the direction of the CNOT gate to satisfy constraints
                # given by the coupling map
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", q0)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", q1)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                g["qargs"] = [g["qargs"][1], g["qargs"][0]]
                gates_fixed += [g]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", q0)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
                gates_fixed += [
                    {
                        "type": "op",
                        "name": "u2",
                        "qargs": [("q", q1)],
                        "cargs": [],
                        "params": [sympify(0), sympify(np.pi)],
                        "condition": None,
                    }
                ]
            elif (g["qargs"][0][1], g["qargs"][1][1]) == (q0, q1):
                gates_fixed += [g]
        else:
            gates_fixed += [g]

    # combine consecutive single qubit gates and simplify the result
    u0 = np.identity(2)
    u0_gates = []
    u1 = np.identity(2)
    u1_gates = []
    for g in gates_fixed:
        if g["name"] == "cx":
            # add single qubit gates to the compiled circuit
            if not np.array_equal(u0, np.identity(2)):
                try:
                    theta, phi, lamb, tmp = euler_angles_1q(u0)
                    name, params, qasm = simplify_U(theta, phi, lamb)
                    if name == "u3":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", q0)],
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
                            [("q", q0)],
                            [],
                            [sympify(params[1]), sympify(params[2])],
                            None,
                        )
                    elif name == "u1":
                        compiled_dag.apply_operation_back(
                            name, [("q", q0)], [], [sympify(params[2])], None
                        )
                except Exception as e:
                    # fallback if decomposition provided by qiskit fails
                    for gg in u0_gates:
                        compiled_dag.apply_operation_back(
                            gg["name"],
                            gg["qargs"],
                            gg["cargs"],
                            gg["params"],
                            gg["condition"],
                        )
                u0 = np.identity(2)
                u0_gates = []
            if not np.array_equal(u1, np.identity(2)):
                try:
                    theta, phi, lamb, tmp = euler_angles_1q(u1)
                    name, params, qasm = simplify_U(theta, phi, lamb)
                    if name == "u3":
                        compiled_dag.apply_operation_back(
                            name,
                            [("q", q1)],
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
                            [("q", q1)],
                            [],
                            [sympify(params[1]), sympify(params[2])],
                            None,
                        )
                    elif name == "u1":
                        compiled_dag.apply_operation_back(
                            name, [("q", q1)], [], [sympify(params[2])], None
                        )
                except Exception as e:
                    # fallback if decomposition provided by qiskit fails
                    for gg in u1_gates:
                        compiled_dag.apply_operation_back(
                            gg["name"],
                            gg["qargs"],
                            gg["cargs"],
                            gg["params"],
                            gg["condition"],
                        )
                u1 = np.identity(2)
                u1_gates = []
            compiled_dag.apply_operation_back(
                g["name"], g["qargs"], g["cargs"], g["params"], g["condition"]
            )

        else:
            # combine single gates
            if g["qargs"][0][1] == q0:
                u0 = np.dot(single_gate_matrix(g["name"], g["params"]), u0)
                u0_gates += [g]
            else:
                u1 = np.dot(single_gate_matrix(g["name"], g["params"]), u1)
                u1_gates += [g]

    # add single qubit gates to the compiled circuit
    if not np.array_equal(u0, np.identity(2)):
        try:
            theta, phi, lamb, tmp = euler_angles_1q(u0)
            name, params, qasm = simplify_U(theta, phi, lamb)
            if name == "u3":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", q0)],
                    [],
                    [sympify(params[0]), sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u2":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", q0)],
                    [],
                    [sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u1":
                compiled_dag.apply_operation_back(
                    name, [("q", q0)], [], [sympify(params[2])], None
                )
        except Exception as e:
            # fallback if decomposition provided by qiskit fails
            for gg in u0_gates:
                compiled_dag.apply_operation_back(
                    gg["name"], gg["qargs"], gg["cargs"], gg["params"], gg["condition"]
                )
        u0 = np.identity(2)
        u0_gates = []
    if not np.array_equal(u1, np.identity(2)):
        try:
            theta, phi, lamb, tmp = euler_angles_1q(u1)
            name, params, qasm = simplify_U(theta, phi, lamb)
            if name == "u3":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", q1)],
                    [],
                    [sympify(params[0]), sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u2":
                compiled_dag.apply_operation_back(
                    name,
                    [("q", q1)],
                    [],
                    [sympify(params[1]), sympify(params[2])],
                    None,
                )
            elif name == "u1":
                compiled_dag.apply_operation_back(
                    name, [("q", q1)], [], [sympify(params[2])], None
                )
        except Exception as e:
            # fallback if decomposition provided by qiskit fails
            for gg in u1_gates:
                compiled_dag.apply_operation_back(
                    gg["name"], gg["qargs"], gg["cargs"], gg["params"], gg["condition"]
                )
        u1 = np.identity(2)
        u1_gates = []
    # return compiled circuit
    return compiled_dag


# optimize groups of gates and add them to the compiled circuit
def optimize_gate_groups(grouped_gates, coupling_map, empty_dag, gate_costs):

    from sympy import sympify

    compiled_dag = empty_dag

    import networkx as nx

    gate_groups = nx.topological_sort(grouped_gates)

    # iterate over all gates in the group and build corresponding 4x4 unitary matrix
    for index in gate_groups:
        group = grouped_gates.node[index]
        if len(group["qubits"]) == 2:
            cost_before = cost_of_group(group["gates"], gate_costs)
            q = [group["qubits"][0], group["qubits"][1]]
            if (("q", q[0]), ("q", q[1])) in coupling_map:
                q = [group["qubits"][0], group["qubits"][1]]
            else:
                if (("q", q[1]), ("q", q[0])) in coupling_map:
                    q = [group["qubits"][1], group["qubits"][0]]

            # estimate whether KAK decomposition (which is time intensive) will improve the result
            if count_cx_gates(group["gates"]) <= 3:
                compiled_dag = apply_gates(compiled_dag, group["gates"], q[0], q[1])
                continue

            U = np.identity(4)
            cnot1 = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
            )
            cnot2 = np.array(
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
            )
            for g in group["gates"]:
                if g["name"] == "cx":
                    if g["qargs"][0][1] == q[0]:
                        U = np.dot(cnot1, U)
                    else:
                        U = np.dot(cnot2, U)
                else:
                    if g["qargs"][0][1] == q[0]:
                        U = np.dot(
                            np.kron(
                                single_gate_matrix(g["name"], g["params"]),
                                np.identity(2),
                            ),
                            U,
                        )
                    else:
                        U = np.dot(
                            np.kron(
                                np.identity(2),
                                single_gate_matrix(g["name"], g["params"]),
                            ),
                            U,
                        )

            new_gates = []
            try:
                for gate in two_qubit_kak(U):
                    i0 = gate["args"][0]
                    if gate["name"] == "cx":
                        i1 = gate["args"][1]
                        new_gates += [
                            {
                                "type": "op",
                                "name": "cx",
                                "qargs": [("q", q[i0]), ("q", q[i1])],
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
                                "qargs": [("q", q[i0])],
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
                                "qargs": [("q", q[i0])],
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
                                "qargs": [("q", q[i0])],
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
            except Exception as e:
                cost_after = cost_before + 1
                pass
            if cost_after < cost_before:
                # add gates in the decomposition
                compiled_dag = apply_gates(compiled_dag, new_gates, q[0], q[1])
            else:
                # add original gates
                compiled_dag = apply_gates(compiled_dag, group["gates"], q[0], q[1])
        else:
            # apply measurement gates and barrier gates without optimization
            for g in group["gates"]:
                compiled_dag.apply_operation_back(
                    g["name"], g["qargs"], g["cargs"], g["params"], g["condition"]
                )

    return compiled_dag
