from typing import Mapping, List, Tuple

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.passes.mapping.size import SizeMapper, Reg, ArchNode


class SimpleSizeMapper(SizeMapper[Reg, ArchNode]):
    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """Perform a simple greedy mapping of the cheapest gate to the architecture."""

        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.

        def simple_saved_gates(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> int:
            """We have to repackage the second argument of place into an iterable."""
            return self.saved_gates((place[0], [place[1]]))

        return self._inner_simple(binops, current_mapping, self.arch_graph,
                                  simple_saved_gates)[0].mapped_to