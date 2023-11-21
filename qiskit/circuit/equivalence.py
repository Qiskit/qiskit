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

import copy
from collections import namedtuple

from rustworkx.visualization import graphviz_draw
import rustworkx as rx

from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression

Key = namedtuple("Key", ["name", "num_qubits"])
Equivalence = namedtuple("Equivalence", ["params", "circuit"])  # Ordered to match Gate.params
NodeData = namedtuple("NodeData", ["key", "equivs"])
EdgeData = namedtuple("EdgeData", ["index", "num_gates", "rule", "source"])


class EquivalenceLibrary:
    """A library providing a one-way mapping of Gates to their equivalent
    implementations as QuantumCircuits."""

    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base (Optional[EquivalenceLibrary]):  Base equivalence library to
                be referenced if an entry is not found in this library.
        """
        self._base = base

        if base is None:
            self._graph = rx.PyDiGraph()
            self._key_to_node_index = {}
            self._rule_count = 0
        else:
            self._graph = base._graph.copy()
            self._key_to_node_index = copy.deepcopy(base._key_to_node_index)
            self._rule_count = base._rule_count

    @property
    def graph(self) -> rx.PyDiGraph:
        """Return graph representing the equivalence library data.

        This property should be treated as read-only as it provides
        a reference to the internal state of the :class:`~.EquivalenceLibrary` object.
        If the graph returned by this property is mutated it could corrupt the
        the contents of the object. If you need to modify the output ``PyDiGraph``
        be sure to make a copy prior to any modification.

        Returns:
            PyDiGraph: A graph object with equivalence data in each node.
        """
        return self._graph

    def _set_default_node(self, key):
        """Create a new node if key not found"""
        if key not in self._key_to_node_index:
            self._key_to_node_index[key] = self._graph.add_node(NodeData(key=key, equivs=[]))
        return self._key_to_node_index[key]

    def add_equivalence(self, gate, equivalent_circuit):
        """Add a new equivalence to the library. Future queries for the Gate
        will include the given circuit, in addition to all existing equivalences
        (including those from base).

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            equivalent_circuit (QuantumCircuit): A circuit equivalently
                implementing the given Gate.
        """

        _raise_if_shape_mismatch(gate, equivalent_circuit)
        _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)

        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equiv = Equivalence(params=gate.params.copy(), circuit=equivalent_circuit.copy())

        target = self._set_default_node(key)
        self._graph[target].equivs.append(equiv)

        sources = {
            Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
            for instruction in equivalent_circuit
        }
        edges = [
            (
                self._set_default_node(source),
                target,
                EdgeData(index=self._rule_count, num_gates=len(sources), rule=equiv, source=source),
            )
            for source in sources
        ]
        self._graph.add_edges_from(edges)
        self._rule_count += 1

    def has_entry(self, gate):
        """Check if a library contains any decompositions for gate.

        Args:
            gate (Gate): A Gate instance.

        Returns:
            Bool: True if gate has a known decomposition in the library.
                False otherwise.
        """
        key = Key(name=gate.name, num_qubits=gate.num_qubits)

        return key in self._key_to_node_index

    def set_entry(self, gate, entry):
        """Set the equivalence record for a Gate. Future queries for the Gate
        will return only the circuits provided.

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
                equivalently implementing the given Gate.
        """
        for equiv in entry:
            _raise_if_shape_mismatch(gate, equiv)
            _raise_if_param_mismatch(gate.params, equiv.parameters)

        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equivs = [Equivalence(params=gate.params.copy(), circuit=equiv.copy()) for equiv in entry]

        self._graph[self._set_default_node(key)] = NodeData(key=key, equivs=equivs)

    def get_entry(self, gate):
        """Gets the set of QuantumCircuits circuits from the library which
        equivalently implement the given Gate.

        Parameterized circuits will have their parameters replaced with the
        corresponding entries from Gate.params.

        Args:
            gate (Gate) - Gate: A Gate instance.

        Returns:
            List[QuantumCircuit]: A list of equivalent QuantumCircuits. If empty,
                library contains no known decompositions of Gate.

                Returned circuits will be ordered according to their insertion in
                the library, from earliest to latest, from top to base. The
                ordering of the StandardEquivalenceLibrary will not generally be
                consistent across Qiskit versions.
        """
        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        query_params = gate.params

        return [_rebind_equiv(equiv, query_params) for equiv in self._get_equivalences(key)]

    def keys(self):
        """Return list of keys to key to node index map.

        Returns:
            List: Keys to the key to node index map.
        """
        return self._key_to_node_index.keys()

    def node_index(self, key):
        """Return node index for a given key.

        Args:
            key (Key): Key to an equivalence.

        Returns:
            Int: Index to the node in the graph for the given key.
        """
        return self._key_to_node_index[key]

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
        for key in self._key_to_node_index:
            name, num_qubits = key
            equivalences = self._get_equivalences(key)

            basis = frozenset([f"{name}/{num_qubits}"])
            for params, decomp in equivalences:
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

                label = "{}\n{}".format(str(params), str(decomp) if num_qubits <= 5 else "...")
                graph.add_edge(
                    node_map[basis],
                    node_map[decomp_basis],
                    {"label": label, "fontname": "Courier", "fontsize": str(8)},
                )

        return graph

    def _get_equivalences(self, key):
        """Get all the equivalences for the given key"""
        return (
            self._graph[self._key_to_node_index[key]].equivs
            if key in self._key_to_node_index
            else []
        )


def _raise_if_param_mismatch(gate_params, circuit_parameters):
    gate_parameters = [p for p in gate_params if isinstance(p, ParameterExpression)]

    if set(gate_parameters) != circuit_parameters:
        raise CircuitError(
            "Cannot add equivalence between circuit and gate "
            "of different parameters. Gate params: {}. "
            "Circuit params: {}.".format(gate_parameters, circuit_parameters)
        )


def _raise_if_shape_mismatch(gate, circuit):
    if gate.num_qubits != circuit.num_qubits or gate.num_clbits != circuit.num_clbits:
        raise CircuitError(
            "Cannot add equivalence between circuit and gate "
            "of different shapes. Gate: {} qubits and {} clbits. "
            "Circuit: {} qubits and {} clbits.".format(
                gate.num_qubits, gate.num_clbits, circuit.num_qubits, circuit.num_clbits
            )
        )


def _rebind_equiv(equiv, query_params):
    equiv_params, equiv_circuit = equiv
    param_map = {x: y for x, y in zip(equiv_params, query_params) if isinstance(x, Parameter)}
    equiv = equiv_circuit.assign_parameters(param_map, inplace=False, flat_input=True)

    return equiv
