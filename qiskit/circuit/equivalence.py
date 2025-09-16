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

"""Gate equivalence library."""

from rustworkx.visualization import graphviz_draw
import rustworkx as rx


from qiskit.exceptions import InvalidFileError
from qiskit._accelerate.equivalence import (  # pylint: disable=unused-import
    BaseEquivalenceLibrary,
    Key,
    Equivalence,
    NodeData,
    EdgeData,
)


class EquivalenceLibrary(BaseEquivalenceLibrary):
    """A library providing a one-way mapping of Gates to their equivalent
    implementations as QuantumCircuits."""

    def draw(self, filename=None):
        """Draws the equivalence relations available in the library.

        Args:
            filename (str): An optional path to write the output image to
                if specified this method will return None.

        Returns:
            PIL.Image or IPython.display.SVG: Drawn equivalence library as an
                IPython SVG if in a jupyter notebook, or as a PIL.Image otherwise.

        Raises:
            InvalidFileError: if filename is not valid.
        """
        image_type = None
        if filename:
            if "." not in filename:
                raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
            image_type = filename.split(".")[-1]
        return graphviz_draw(
            self._build_basis_graph(),
            lambda node: {"label": node["label"]},
            lambda edge: edge,
            filename=filename,
            image_type=image_type,
        )

    def _build_basis_graph(self):
        graph = rx.PyDiGraph()

        node_map = {}
        for key in super().keys():
            name, num_qubits = key.name, key.num_qubits
            equivalences = self._get_equivalences(key)

            basis = frozenset([f"{name}/{num_qubits}"])
            for equivalence in equivalences:
                params, decomp = equivalence.params, equivalence.circuit
                decomp_basis = frozenset(
                    f"{name}/{num_qubits}"
                    for name, num_qubits in {
                        (instruction.operation.name, instruction.operation.num_qubits)
                        for instruction in decomp.data
                    }
                )
                if basis not in node_map:
                    basis_node = graph.add_node({"basis": basis, "label": str(set(basis))})
                    node_map[basis] = basis_node
                if decomp_basis not in node_map:
                    decomp_basis_node = graph.add_node(
                        {"basis": decomp_basis, "label": str(set(decomp_basis))}
                    )
                    node_map[decomp_basis] = decomp_basis_node

                label = f"{str(params)}\n{str(decomp) if num_qubits <= 5 else '...'}"
                graph.add_edge(
                    node_map[basis],
                    node_map[decomp_basis],
                    {"label": label, "fontname": "Courier", "fontsize": str(8)},
                )

        return graph
