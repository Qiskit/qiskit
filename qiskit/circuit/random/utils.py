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


def random_circuit_from_graph(
    interaction_graph,
    min_2q_gate_per_edge,
    max_operands=2,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
    insert_1q_oper=False,
    prob_conditional: float = 0.1,
):
    """Generate random circuit of arbitrary size and form which strictly respects
    the interaction graph passed as argument.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`.
    User can attach a float value indicating the probability of getting selected as a metadata to
    the edge of the graph generated. If all the probabilities is passed as `None`, then the probability
    of each qubit-pair of getting selected is set to 1/N.
    (where N is the number of edges in the interaction_graph passed in)

    If float values are present as probabilities but some are None, this will raise a ValueError.

    If :arg:`max_operands` is set to 2, then a 2Q gate is chosen at random and a qubit-pair is
    also chosen at random based on the probability attached to the qubit-pair, this 2Q gate
    is applied to that qubit-pair and the rest of the qubits are filled with 1Q operations
    based on :arg:`insert_1q_oper` is set to True. This makes a single layer of the cirucit.

    Example:

    .. plot::
       :include-source:

       from qiskit.circuit.random.utils import random_circuit_from_graph
       import rustworkx as rx
       pydi_graph = rx.PyDiGraph()
       pydi_graph.add_nodes_from(range(8))
       cp_map = [(0, 2, 0.18), (1, 3, 0.15), (2, 4, 0.15), (3, 4, 0.22), (5, 7, 0.13), (4, 7, 0.17)]
       pydi_graph.add_edges_from(cp_map)
       inter_graph = (pydi_graph, None, None, None)
       qc = random_circuit_from_graph(inter_graph, max_num_qubit_usage = 3, measure = True)
       qc.draw(output='mpl')

    Args:
        interaction_graph (int): Interaction Graph
        min_2q_gate_per_edge (int): Minimum number of times every qubit-pair must be used.
        max_operands (int): maximum qubit operands of each gate (between 1 and 2)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets. ( insert_1q_oper should be true )
        seed (int): sets random seed (optional)
        insert_1q_oper (bool): Insert 1Q operations to the qubits which are left after
        applying a 2Q gate on the selected qubit-pair.
        prob_conditional (float): Probability less than 1.0, this is used to control the
        occurence of conditionals in the circuit.

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
        ValueError: when the probabilities for qubit-pairs are passed and they do not add to 1.0.
        ValueError: when some edges have probability as None.
    """
    # Leaving out 'free_nodes'
    pydi_graph, pydi_graph_node_map, _, _ = interaction_graph
    edges_probs = pydi_graph.edges()
    num_edges = pydi_graph.num_edges()

    # Just a switch for the probability weighted selection of a particular qubit-pair.
    prob_weighted_mapping = False

    # If all the values of the probability is None, then all the values of the probability
    # is assumed to have a uniform distrubution.
    # If any of the values of the probability is None, then it would raise an error.
    if all(edges_probs):
        prob_weighted_mapping = True
    elif any(edges_probs):
        raise ValueError("The probabilities of a qubit-pair getting seleted contains `None`")
    elif None in edges_probs:
        prob_weighted_mapping = True
        edges_probs = [1.0 / num_edges for _ in range(num_edges)]

    # edge weights in interaction_graph must be positive
    if prob_weighted_mapping:
        for prob in edges_probs:
            if prob < 0:
                raise ValueError("The probability should be positive")

    # edge weights in interaction_graph should be normalized to sum up to 1.0
    if prob_weighted_mapping and not np.isclose(np.sum(edges_probs), 1.000, rtol=0.001):
        raise ValueError(
            "The sum of all the probabilities of a qubit-pair to be selected is not 1.0"
        )

    if pydi_graph_node_map is not None:
        # The number of vertices in interaction_graph define the number of qubits
        # in the circuit to be generated
        qubits = list(pydi_graph_node_map.keys())
    else:
        n_q = pydi_graph.num_nodes()
        qubits = [Qubit(QuantumRegister(n_q, "q"), idx) for idx in pydi_graph.nodes()]
    num_qubits = len(qubits)

    if num_qubits == 0:
        return QuantumCircuit()

    if max_operands < 1 or max_operands > 2:
        raise CircuitError("max_operands must be between 1 and 2")

    max_operands = max_operands if num_qubits > max_operands else num_qubits

    edge_list = list(pydi_graph.edge_list())

    if num_edges == 0 or max_operands == 1:
        # If there is no edge then there could be no 2Q operation
        # or, if only 1Q operations are allowed, then no point of edges.
        return random_circuit(
            num_qubits=num_qubits,
            depth=min_2q_gate_per_edge,
            max_operands=1,
            measure=measure,
            conditional=conditional,
            reset=reset,
            seed=seed,
        )

    gates_1q, gates_2q, _, _ = _get_gates()

    gates_2q = np.array(
        gates_2q, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )

    # Reset is added if 1Q operation is allowed and arg: `reset` is set to true
    if insert_1q_oper:
        if reset:
            gates_1q.append((Reset, 1, 0))

        gates_1q = np.array(
            gates_1q, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
        )

    qc = QuantumCircuit(qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    def _get_extra_1q_oper(extra_1q_gates, parameters):
        current_gate = extra_1q_gates[0]
        extra_1q_gates = extra_1q_gates[1:]
        num_params = current_gate["num_params"]
        params = parameters[:num_params]
        parameters = parameters[num_params:]
        return current_gate["class"](*params)

    stop = False
    layer_idx = 0
    edges_used = {edge: 0 for edge in edge_list}

    while not stop:
        # This outer loop will keep on applying gates to qubits until every qubit-pair
        # has a 2Q operation applied `min_2q_gate_per_edge`.

        qubit_idx_used = set()
        qubit_idx_not_used = set(range(num_qubits))

        # normalized edge weights represent the probability with which each qubit-pair
        # interaction is inserted into a layer.
        edge_choices = rng.choice(
            edge_list,
            size=num_edges,
            replace=True,
            p=edges_probs if prob_weighted_mapping else None,
        )
        gate_choices = rng.choice(gates_2q, size=num_edges, replace=True)

        # Some extra 1Q Gate in case we need to fill an empty qubit
        if insert_1q_oper:
            extra_1q_gates = rng.choice(gates_1q, size=num_qubits, replace=True)
            cumsum_params = np.cumsum(extra_1q_gates["num_params"], dtype=np.int64)
            parameters_1q = rng.uniform(0, 2 * np.pi, size=cumsum_params[-1])

        cumsum_params = np.cumsum(gate_choices["num_params"], dtype=np.int64)
        parameters = rng.uniform(0, 2 * np.pi, size=cumsum_params[-1])

        if conditional and layer_idx != 0:

            is_conditional = rng.random(size=len(gate_choices)) < prob_conditional
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0

            for current_gate, num_gate_params, edge, is_cond in zip(
                gate_choices["class"],
                gate_choices["num_params"],
                edge_choices,
                is_conditional,
            ):
                params = parameters[:num_gate_params]
                parameters = parameters[num_gate_params:]
                operation = current_gate(*params)
                control_qubit, target_qubit = tuple(edge)

                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation = operation.c_if(cr, int(condition_values[c_ptr]))
                    c_ptr += 1

                if control_qubit in qubit_idx_used and target_qubit not in qubit_idx_used:
                    if insert_1q_oper:
                        qc._append(
                            CircuitInstruction(
                                operation=_get_extra_1q_oper(extra_1q_gates, parameters_1q),
                                qubits=[qubits[target_qubit]],
                            )
                        )
                        qubit_idx_used.update({target_qubit})
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                elif control_qubit not in qubit_idx_used and target_qubit in qubit_idx_used:
                    if insert_1q_oper:
                        qc._append(
                            CircuitInstruction(
                                operation=_get_extra_1q_oper(extra_1q_gates, parameters_1q),
                                qubits=[qubits[control_qubit]],
                            )
                        )
                        qubit_idx_used.update({control_qubit})
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
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
                    edges_used[(control_qubit, target_qubit)] += 1

        else:
            for current_gate, num_gate_params, edge in zip(
                gate_choices["class"],
                gate_choices["num_params"],
                edge_choices,
            ):
                params = parameters[:num_gate_params]
                parameters = parameters[num_gate_params:]
                operation = current_gate(*params)
                control_qubit, target_qubit = tuple(edge)

                if control_qubit in qubit_idx_used and target_qubit not in qubit_idx_used:
                    if insert_1q_oper:
                        qc._append(
                            CircuitInstruction(
                                operation=_get_extra_1q_oper(extra_1q_gates, parameters_1q),
                                qubits=[qubits[target_qubit]],
                            )
                        )
                        qubit_idx_used.update({target_qubit})
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
                elif control_qubit not in qubit_idx_used and target_qubit in qubit_idx_used:
                    if insert_1q_oper:
                        qc._append(
                            CircuitInstruction(
                                operation=_get_extra_1q_oper(extra_1q_gates, parameters_1q),
                                qubits=[qubits[control_qubit]],
                            )
                        )
                        qubit_idx_used.update({control_qubit})
                        qubit_idx_not_used = qubit_idx_not_used - qubit_idx_used
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
                    edges_used[(control_qubit, target_qubit)] += 1

        layer_idx += 1

        # check if every edge has been used at-least `min_2q_gate_per_edge` number of times.
        reached_depth = np.array(list(edges_used.values())) >= min_2q_gate_per_edge
        if all(reached_depth):
            stop = True

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def random_circuit(
    num_qubits,
    depth,
    max_operands=4,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
    num_operand_distribution: dict = None,
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
        num_operand_distribution (dict): a distribution of gates that specifies the ratio
        of 1-qubit, 2-qubit, 3-qubit, ..., n-qubit gates in the random circuit. Expect a
        deviation from the specified ratios that depends on the size of the requested
        random circuit. (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    if num_operand_distribution:
        if min(num_operand_distribution.keys()) < 1 or max(num_operand_distribution.keys()) > 4:
            raise CircuitError("'num_operand_distribution' must have keys between 1 and 4")
        for key, prob in num_operand_distribution.items():
            if key > num_qubits and prob != 0.0:
                raise CircuitError(
                    f"'num_operand_distribution' cannot have {key}-qubit gates"
                    f" for circuit with {num_qubits} qubits"
                )
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    if not num_operand_distribution and max_operands:
        if max_operands < 1 or max_operands > 4:
            raise CircuitError("max_operands must be between 1 and 4")
        max_operands = max_operands if num_qubits > max_operands else num_qubits
        rand_dist = rng.dirichlet(
            np.ones(max_operands)
        )  # This will create a random distribution that sums to 1
        num_operand_distribution = {i + 1: rand_dist[i] for i in range(max_operands)}
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    # Here we will use np.isclose() because very rarely there might be floating
    # point precision errors
    if not np.isclose(sum(num_operand_distribution.values()), 1):
        raise CircuitError("The sum of all the values in 'num_operand_distribution' is not 1.")

    if num_qubits == 0:
        return QuantumCircuit()

    gates_1q, gates_2q, gates_3q, gates_4q = _get_gates()

    if reset:
        gates_1q.append((Reset, 1, 0))

    gates_1q = np.array(
        gates_1q, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )
    gates_2q = np.array(gates_2q, dtype=gates_1q.dtype)
    gates_3q = np.array(gates_3q, dtype=gates_1q.dtype)
    gates_4q = np.array(gates_4q, dtype=gates_1q.dtype)

    all_gate_lists = [gates_1q, gates_2q, gates_3q, gates_4q]

    # Here we will create a list 'gates_to_consider' that will have a
    # subset of different n-qubit gates and will also create a list for
    # ratio (or probability) for each gates
    gates_to_consider = []
    distribution = []
    for n_qubits, ratio in num_operand_distribution.items():
        gate_list = all_gate_lists[n_qubits - 1]
        gates_to_consider.extend(gate_list)
        distribution.extend([ratio / len(gate_list)] * len(gate_list))

    gates = np.array(gates_to_consider, dtype=gates_1q.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Counter to keep track of number of different gate types
    counter = np.zeros(len(all_gate_lists) + 1, dtype=np.int64)
    total_gates = 0

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.
        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.

        # Due to the stochastic nature of generating a random circuit, the resulting ratios
        # may not precisely match the specified values from `num_operand_distribution`. Expect
        # greater deviations from the target ratios in quantum circuits with fewer qubits and
        # shallower depths, and smaller deviations in larger and deeper quantum circuits.
        # For more information on how the distribution changes with number of qubits and depth
        # refer to the pull request #12483 on Qiskit GitHub.

        gate_specs = rng.choice(gates, size=len(qubits), p=distribution)
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)

        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]

        slack = num_qubits - cumulative_qubits[max_index - 1]

        # Updating the counter for 1-qubit, 2-qubit, 3-qubit and 4-qubit gates
        gate_qubits = gate_specs["num_qubits"]
        counter += np.bincount(gate_qubits, minlength=len(all_gate_lists) + 1)

        total_gates += len(gate_specs)

        # Slack handling loop, this loop will add gates to fill
        # the slack while respecting the 'num_operand_distribution'
        while slack > 0:
            gate_added_flag = False

            for key, dist in sorted(num_operand_distribution.items(), reverse=True):
                if slack >= key and counter[key] / total_gates < dist:
                    gate_to_add = np.array(
                        all_gate_lists[key - 1][rng.integers(0, len(all_gate_lists[key - 1]))]
                    )
                    gate_specs = np.hstack((gate_specs, gate_to_add))
                    counter[key] += 1
                    total_gates += 1
                    slack -= key
                    gate_added_flag = True
            if not gate_added_flag:
                break

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
