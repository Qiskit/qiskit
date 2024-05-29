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

from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    CircuitInstruction,
    Qubit,
    QuantumRegister,
)
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError


def _get_gates():
    gates_1q = [
        # (Gate class, number of qubits, number of parameters)
        (standard_gates.IGate, 1, 0),
        (standard_gates.SXGate, 1, 0),
        (standard_gates.XGate, 1, 0),
        (standard_gates.RZGate, 1, 1),
        (standard_gates.RGate, 1, 2),
        (standard_gates.HGate, 1, 0),
        (standard_gates.PhaseGate, 1, 1),
        (standard_gates.RXGate, 1, 1),
        (standard_gates.RYGate, 1, 1),
        (standard_gates.SGate, 1, 0),
        (standard_gates.SdgGate, 1, 0),
        (standard_gates.SXdgGate, 1, 0),
        (standard_gates.TGate, 1, 0),
        (standard_gates.TdgGate, 1, 0),
        (standard_gates.UGate, 1, 3),
        (standard_gates.U1Gate, 1, 1),
        (standard_gates.U2Gate, 1, 2),
        (standard_gates.U3Gate, 1, 3),
        (standard_gates.YGate, 1, 0),
        (standard_gates.ZGate, 1, 0),
    ]

    gates_2q = [
        (standard_gates.CXGate, 2, 0),
        (standard_gates.DCXGate, 2, 0),
        (standard_gates.CHGate, 2, 0),
        (standard_gates.CPhaseGate, 2, 1),
        (standard_gates.CRXGate, 2, 1),
        (standard_gates.CRYGate, 2, 1),
        (standard_gates.CRZGate, 2, 1),
        (standard_gates.CSXGate, 2, 0),
        (standard_gates.CUGate, 2, 4),
        (standard_gates.CU1Gate, 2, 1),
        (standard_gates.CU3Gate, 2, 3),
        (standard_gates.CYGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        (standard_gates.RXXGate, 2, 1),
        (standard_gates.RYYGate, 2, 1),
        (standard_gates.RZZGate, 2, 1),
        (standard_gates.RZXGate, 2, 1),
        (standard_gates.XXMinusYYGate, 2, 2),
        (standard_gates.XXPlusYYGate, 2, 2),
        (standard_gates.ECRGate, 2, 0),
        (standard_gates.CSGate, 2, 0),
        (standard_gates.CSdgGate, 2, 0),
        (standard_gates.SwapGate, 2, 0),
        (standard_gates.iSwapGate, 2, 0),
    ]
    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
        (standard_gates.CSwapGate, 3, 0),
        (standard_gates.CCZGate, 3, 0),
        (standard_gates.RCCXGate, 3, 0),
    ]
    gates_4q = [
        (standard_gates.C3SXGate, 4, 0),
        (standard_gates.RC3XGate, 4, 0),
    ]
    return (gates_1q, gates_2q, gates_3q, gates_4q)


def random_circuit_with_graph(
    interaction_graph,
    max_num_qubit_usage,
    max_operands=2,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
):
    """Generate random circuit of arbitrary size and form which strictly respects
    the interaction graph passed as argument.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`.
    User can attach a float value indicating the probability of getting selected as a metadata to
    the edge of the graph generated. Pass in None if the probability is not available.
    Even if a single probability is None, this will turn off the probabilistic selection of qubit-pair.
    Example:

    .. plot::
       :include-source:

       from qiskit.circuit.random.utils import random_circuit_with_graph
       import rustworkx as rx
       pydi_graph = rx.PyDiGraph()
       pydi_graph.add_nodes_from(range(8))
       cp_map = [(0, 2, 0.18), (1, 3, 0.15), (2, 4, 0.15), (3, 4, 0.22), (5, 7, 0.13), (4, 7, 0.17)]
       pydi_graph.add_edges_from(cp_map)
       inter_graph = (pydi_graph, None, None, None)
       qc = random_circuit_with_graph(inter_graph, max_num_qubit_usage = 3, measure = True)
       qc.draw(output='mpl')

    Args:
        interaction_graph (int): Interaction Graph
        max_num_qubit_usage (int): Maximum number of times a single qubit can be used in total.
        max_operands (int): maximum qubit operands of each gate (between 1 and 2)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
        ValueError: when the probabilities for qubit-pairs are passed and they do not add to 1.0
    """
    pydi_graph, pydi_graph_node_map, _, _ = interaction_graph
    edges_probs = pydi_graph.edges()

    # Just a switch for the probability weighted selection of a particular qubit-pair
    # If any value of the probability is None or the whole probability list is None,
    # then, the probability weighted selection of qubit-pair would be turned off.
    prob_weighted_mapping = not None in edges_probs

    if prob_weighted_mapping and not np.isclose(np.sum(edges_probs), 1.000, rtol=0.001):
        raise ValueError(
            "The sum of all the probabilities of a qubit-pair to be selected is not 1.0"
        )

    if pydi_graph_node_map is not None:
        qubits = list(pydi_graph_node_map.keys())
    else:
        n_q = pydi_graph.num_nodes()
        qubits = [Qubit(QuantumRegister(n_q, "q"), idx) for idx in pydi_graph.nodes()]
    num_qubits = len(qubits)

    if num_qubits == 0:
        return QuantumCircuit()

    if pydi_graph.edges() == []:
        return random_circuit(
            num_qubits=num_qubits,
            depth=2,
            max_operands=1,
            measure=measure,
            conditional=conditional,
            reset=reset,
            seed=seed,
        )

    if max_operands < 1 or max_operands > 2:
        raise CircuitError("max_operands must be between 1 and 2")
    max_operands = max_operands if num_qubits > max_operands else num_qubits
    gates_1q, gates_2q, _, _ = _get_gates()

    if reset:
        gates_1q.append((Reset, 1, 0))

    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)

    gates = np.array(
        gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)
    edge_prob_map = None

    edge_prob_map = {
        "edge": pydi_graph.edge_list(),
        "prob": edges_probs if prob_weighted_mapping else None,
    }

    def _get_extra_1q_oper(extra_1q_gates, parameters):
        current_gate = extra_1q_gates[0]
        extra_1q_gates = extra_1q_gates[1:]
        num_params = current_gate["num_params"]
        params = parameters[:num_params]
        parameters = parameters[num_params:]
        return current_gate["class"](*params)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_idx in range(max_num_qubit_usage):
        qubit_idx_used = set()
        qubit_idx_not_used = set(range(num_qubits))

        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_choices = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_choices["num_qubits"], dtype=np.int64)

        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_choices = gate_choices[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_choices = np.hstack((gate_choices, rng.choice(gates_1q, size=slack)))

        # Some extra 1Q Gate in case we need to fill an empty qubit
        extra_1q_gates = rng.choice(gates_1q, size=num_qubits)

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.

        cumsum_params = np.cumsum(gate_choices["num_params"], dtype=np.int64)
        # Get some extra parameters in case we need them to pair with the extra_1q_gates
        parameters = rng.uniform(0, 2 * np.pi, size=cumsum_params[-1] + num_qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_idx != 0:

            is_conditional = rng.random(size=len(gate_choices)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for current_gate, num_gate_qubits, num_gate_params, is_cond in zip(
                gate_choices["class"],
                gate_choices["num_qubits"],
                gate_choices["num_params"],
                is_conditional,
            ):
                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation = operation.c_if(cr, int(condition_values[c_ptr]))
                    c_ptr += 1

                params = parameters[:num_gate_params]
                parameters = parameters[num_gate_params:]
                operation = current_gate(*params)
                if num_gate_qubits == 1:
                    use_qubit_idx = qubit_idx_not_used.pop()
                    qc._append(
                        CircuitInstruction(operation=operation, qubits=[qubits[use_qubit_idx]])
                    )
                    qubit_idx_used.update({use_qubit_idx})
                    qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                elif num_gate_qubits == 2:
                    for edge in rng.choice(
                        edge_prob_map["edge"],
                        size=1,
                        replace=False,
                        p=edge_prob_map["prob"],
                    ):
                        edge = tuple(edge)
                        control_qubit, target_qubit = edge

                        if control_qubit in qubit_idx_used and target_qubit not in qubit_idx_used:
                            qc._append(
                                CircuitInstruction(
                                    operation=_get_extra_1q_oper(extra_1q_gates, parameters),
                                    qubits=[qubits[target_qubit]],
                                )
                            )
                        elif control_qubit not in qubit_idx_used and target_qubit in qubit_idx_used:
                            qc._append(
                                CircuitInstruction(
                                    operation=_get_extra_1q_oper(extra_1q_gates, parameters),
                                    qubits=[qubits[control_qubit]],
                                )
                            )
                        elif control_qubit in qubit_idx_used and target_qubit in qubit_idx_used:
                            continue

                        else:
                            qc._append(
                                CircuitInstruction(
                                    operation=operation,
                                    qubits=[qubits[control_qubit], qubits[target_qubit]],
                                )
                            )

                        qubit_idx_used.update(set(edge))
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                        break

        else:
            for current_gate, num_gate_qubits, num_gate_params in zip(
                gate_choices["class"], gate_choices["num_qubits"], gate_choices["num_params"]
            ):
                params = parameters[:num_gate_params]
                parameters = parameters[num_gate_params:]
                operation = current_gate(*params)
                if num_gate_qubits == 1:
                    use_qubit_idx = qubit_idx_not_used.pop()
                    qc._append(
                        CircuitInstruction(operation=operation, qubits=[qubits[use_qubit_idx]])
                    )
                    qubit_idx_used.update({use_qubit_idx})
                    qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                elif num_gate_qubits == 2:
                    for edge in rng.choice(
                        edge_prob_map["edge"],
                        size=1,
                        replace=False,
                        p=edge_prob_map["prob"],
                    ):
                        edge = tuple(edge)
                        control_qubit, target_qubit = edge

                        if control_qubit in qubit_idx_used and target_qubit not in qubit_idx_used:
                            qc._append(
                                CircuitInstruction(
                                    operation=_get_extra_1q_oper(extra_1q_gates, parameters),
                                    qubits=[qubits[target_qubit]],
                                )
                            )
                        elif control_qubit not in qubit_idx_used and target_qubit in qubit_idx_used:
                            qc._append(
                                CircuitInstruction(
                                    operation=_get_extra_1q_oper(extra_1q_gates, parameters),
                                    qubits=[qubits[control_qubit]],
                                )
                            )
                        elif control_qubit in qubit_idx_used and target_qubit in qubit_idx_used:
                            continue

                        else:
                            qc._append(
                                CircuitInstruction(
                                    operation=operation,
                                    qubits=[qubits[control_qubit], qubits[target_qubit]],
                                )
                            )

                        qubit_idx_used.update(set(edge))
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                        break

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def random_circuit(
    num_qubits, depth, max_operands=4, measure=False, conditional=False, reset=False, seed=None
):
    """Generate random circuit of arbitrary size and form.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :include-source:

       from qiskit.circuit.random import random_circuit

       circ = random_circuit(2, 2, measure=True)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum qubit operands of each gate (between 1 and 4)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if num_qubits == 0:
        return QuantumCircuit()
    if max_operands < 1 or max_operands > 4:
        raise CircuitError("max_operands must be between 1 and 4")
    max_operands = max_operands if num_qubits > max_operands else num_qubits
    gates_1q, gates_2q, gates_3q, gates_4q = _get_gates()

    if reset:
        gates_1q.append((Reset, 1, 0))

    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)
    if max_operands >= 3:
        gates.extend(gates_3q)
    if max_operands >= 4:
        gates.extend(gates_4q)
    gates = np.array(
        gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])
                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation = operation.c_if(cr, int(condition_values[c_ptr]))
                    c_ptr += 1
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def random_clifford_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo-random Clifford circuit.

    This function will generate a Clifford circuit by randomly selecting the chosen amount of Clifford
    gates from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :include-source:

       from qiskit.circuit.random import random_clifford_circuit

       circ = random_clifford_circuit(num_qubits=2, num_gates=6)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires.
        num_gates (int): number of gates in the circuit.
        gates (list[str]): optional list of Clifford gate names to randomly sample from.
            If ``"all"`` (default), use all Clifford gates in the standard library.
        seed (int | np.random.Generator): sets random seed/generator (optional).

    Returns:
        QuantumCircuit: constructed circuit
    """

    gates_1q = ["i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"]
    gates_2q = ["cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"]
    if gates == "all":
        if num_qubits == 1:
            gates = gates_1q
        else:
            gates = gates_1q + gates_2q

    instructions = {
        "i": (standard_gates.IGate(), 1),
        "x": (standard_gates.XGate(), 1),
        "y": (standard_gates.YGate(), 1),
        "z": (standard_gates.ZGate(), 1),
        "h": (standard_gates.HGate(), 1),
        "s": (standard_gates.SGate(), 1),
        "sdg": (standard_gates.SdgGate(), 1),
        "sx": (standard_gates.SXGate(), 1),
        "sxdg": (standard_gates.SXdgGate(), 1),
        "cx": (standard_gates.CXGate(), 2),
        "cy": (standard_gates.CYGate(), 2),
        "cz": (standard_gates.CZGate(), 2),
        "swap": (standard_gates.SwapGate(), 2),
        "iswap": (standard_gates.iSwapGate(), 2),
        "ecr": (standard_gates.ECRGate(), 2),
        "dcx": (standard_gates.DCXGate(), 2),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs, copy=False)

    return circ
