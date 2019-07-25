from typing import Mapping, Set, FrozenSet, Dict

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper


class SimpleDepthMapper(DepthMapper[Reg, ArchNode]):
    def map(self, circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode] = None) -> Mapping[Reg, ArchNode]:
        """
        Try to map as many two-qubit gates to a maximum matching as possible.

        Note: Does not take into account the scoring function, nor the weights on the graph.

        :param circuit: A circuit to execute
        :param arch_graph: The architecture graph,optionally with weights on edges.
            Default weights are 1.
        :param current_mapping: The current mapping of registers to archictecture nodes.
        :return:
        """
        binops = Mapper._binops_circuit(circuit)
        matching: Set[FrozenSet[ArchNode]] = Mapper.construct_matching(self.arch_graph)
        # First assign the two-qubit gates, because they are restricted by the architecture
        mapping: Dict[Reg, ArchNode] = {}
        for binop in binops:
            if matching:
                # pick an available matching and map this operation to that matching
                node0, node1 = matching.pop()
                mapping[binop.qargs[0]] = node0
                mapping[binop.qargs[1]] = node1
            else:
                # no more matchings
                break

        return mapping