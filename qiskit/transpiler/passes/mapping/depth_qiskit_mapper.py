from typing import Callable, Mapping, Iterable, List

import networkx as nx

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.size_qiskit_mapper import QiskitSizeMapper
from qiskit.transpiler.routing import Swap, util


class QiskitDepthMapper(DepthMapper[Reg, ArchNode]):

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]]) -> None:
        super().__init__(arch_graph, arch_permuter)
        self.qiskit_size_mapper: QiskitSizeMapper[Reg, ArchNode] = \
            QiskitSizeMapper(arch_graph.to_directed(as_view=True),
                             # Convert depth permuter to size permuter
                             # This is only used for SimpleSizeMapper,
                             # so it's fine.
                             util.sequential_permuter(arch_permuter))

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        # We call the size_map function to avoid size-based preconditions.
        binops = Mapper._binops_circuit(circuit)
        return self.qiskit_size_mapper.size_map(circuit, current_mapping, binops)