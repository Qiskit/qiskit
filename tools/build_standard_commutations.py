# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Determines a commutation library over the unparameterizable standard gates, i.e. a dictionary for
   each pair of parameterizable standard gates and all qubit overlaps that maps to either True or False,
   depending on the present commutation relation.
"""


import itertools
from functools import lru_cache
from typing import List
from qiskit.circuit.commutation_checker import _get_relative_placement, _order_operations
from qiskit.circuit import Gate, CommutationChecker
import qiskit.circuit.library.standard_gates as stdg
from qiskit.dagcircuit import DAGOpNode


@lru_cache(None)
def _get_unparameterizable_gates() -> List[Gate]:
    """Retrieve a list of non-parmaterized gates with up to 3 qubits, using the python inspection module
    Return:
        A list of non-parameterized gates to be considered in the commutation library
    """
    # These two gates may require a large runtime in later processing steps
    # blocked_types = [C3SXGate, C4XGate]
    gates = list(stdg.get_standard_gate_name_mapping().values())
    return [g for g in gates if len(g.params) == 0]


def _generate_commutation_dict(considered_gates: List[Gate] = None) -> dict:
    """Compute the commutation relation of considered gates

    Args:
        considered_gates List[Gate]: a list of gates between which the commutation should be determined

    Return:
        A dictionary that includes the commutation relation for each
        considered pair of operations and each relative placement

    """
    commutations = {}
    cc = CommutationChecker()
    for gate0 in considered_gates:

        node0 = DAGOpNode(op=gate0, qargs=list(range(gate0.num_qubits)), cargs=[])
        for gate1 in considered_gates:

            # only consider canonical entries
            (
                (
                    first_gate,
                    _,
                    _,
                ),
                (second_gate, _, _),
            ) = _order_operations(gate0, None, None, gate1, None, None)
            if (first_gate, second_gate) != (gate0, gate1) and gate0.name != gate1.name:
                continue

            # enumerate all relative gate placements with overlap between gate qubits
            gate_placements = itertools.permutations(
                range(gate0.num_qubits + gate1.num_qubits - 1), gate0.num_qubits
            )
            gate_pair_commutation = {}
            for permutation in gate_placements:
                permutation_list = list(permutation)
                gate1_qargs = []

                # use idx_non_overlapping qubits to represent qubits on g1 that are not connected to g0
                next_non_overlapping_qubit_idx = gate0.num_qubits
                for i in range(gate1.num_qubits):
                    if i in permutation_list:
                        gate1_qargs.append(permutation_list.index(i))
                    else:
                        gate1_qargs.append(next_non_overlapping_qubit_idx)
                        next_non_overlapping_qubit_idx += 1

                node1 = DAGOpNode(op=gate1, qargs=gate1_qargs, cargs=[])

                # replace non-overlapping qubits with None to act as a key in the commutation library
                relative_placement = _get_relative_placement(node0.qargs, node1.qargs)

                if not gate0.is_parameterized() and not gate1.is_parameterized():
                    # if no gate includes parameters, compute commutation relation using
                    # matrix multiplication
                    op1 = node0.op
                    qargs1 = node0.qargs
                    cargs1 = node0.cargs
                    op2 = node1.op
                    qargs2 = node1.qargs
                    cargs2 = node1.cargs
                    commutation_relation = cc.commute(
                        op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits=4
                    )

                    gate_pair_commutation[relative_placement] = commutation_relation
                else:
                    pass
                    # TODO

            commutations[gate0.name, gate1.name] = gate_pair_commutation
    return commutations


def _simplify_commuting_dict(commuting_dict: dict) -> dict:
    """Compress some of the commutation library entries

    Args:
        commuting_dict (dict): A commutation dictionary
    Return:
        commuting_dict (dict): A commutation dictionary with simplified entries
    """
    # Remove relative placement key if commutation is independent of relative placement
    for ops in commuting_dict.keys():
        gates_commutations = set(commuting_dict[ops].values())
        if len(gates_commutations) == 1:
            commuting_dict[ops] = next(iter(gates_commutations))

    return commuting_dict


def _dump_commuting_dict_as_python(
    commutations: dict, file_name: str = "../_standard_gates_commutations.py"
):
    """Write commutation dictionary as python object to ./qiskit/circuit/_standard_gates_commutations.py.

    Args:
        commutations (dict): a dictionary that includes the commutation relation for
        each considered pair of operations

    """
    with open(file_name, "w") as fp:
        dir_str = "standard_gates_commutations = {\n"
        for k, v in commutations.items():
            if not isinstance(v, dict):
                # pylint: disable-next=consider-using-f-string
                dir_str += '    ("{}", "{}"): {},\n'.format(*k, v)
            else:
                # pylint: disable-next=consider-using-f-string
                dir_str += '    ("{}", "{}"): {{\n'.format(*k)

                for entry_key, entry_val in v.items():
                    dir_str += f"        {entry_key}: {entry_val},\n"

                dir_str += "    },\n"
        dir_str += "}\n"
        fp.write(dir_str.replace("'", ""))


if __name__ == "__main__":
    cgates = [
        g for g in _get_unparameterizable_gates() if g.name not in ["reset", "measure", "delay"]
    ]
    commutation_dict = _generate_commutation_dict(considered_gates=cgates)
    commutation_dict = _simplify_commuting_dict(commutation_dict)
    _dump_commuting_dict_as_python(commutation_dict)
