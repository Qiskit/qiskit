# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A target object represents the minimum set of information the transpiler needs
from a backend
"""

import logging

import retworkx as rx

from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import CouplingError

logger = logging.getLogger(__name__)


class InstructionProperties:
    """A representation of the properties of a gate implementation."""

    __slots__ = ("length", "error", "properties")

    def __init__(self, length=None, error=None, properties=None):
        self.length = length
        self.error = error
        self.properties = properties


class Target:
    """
    A description of gates on a backend. It exists around a central mapping of
    :class:`~qiskit.circuit.Instruction` objects to their properties on the
    device. As a basic example, lets assume your device is two qubits, supports
    :class:`~qiskit.circuit.library.UGate` on both qubits and
    :class:`~qiskit.circuit.library.CXGate` in both directions you would create
    the gate map like::

        from qiskit.transpiler import Target
        from qiskit.circuit.library import UGate, CXGate
        from qiskit.circuit import Parameter

        gmap = Target()
        theta = Parameter('theta')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        u_props = {
            (0,): InstructionProperties(length=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(length=4.52e-8, error=0.00032115),
        }
        gmap.add_gate(UGate(theta, phi, lam), [(0,), (1,)], properties=u_props)
        cx_props = {
            (0,1): InstructionProperties(length=5.23e-7, error=0.00098115),
            (1,0): InstructionProperties(length=4.52e-7, error=0.00132115),
        }
        gmap.add_gate(CXGate(), [(0, 1), (1, 0)], properties=cx_props)

    .. note::

        This class assumes that qubit indices start at 0 and are a contiguous
        set if you want a submapping the bits will need to be reindexed in
        a new object

    .. note::

        This class is designed to be additive only for gates, qargs, and qubits.
        If you need to remove one of these the best option is to iterate over
        an existing object and create a new subset (or use one of the methods
        to do this). The object internally caches different views and these
        would potentially be invalidated by removals.
    """

    __slots__ = (
        "num_qubits",
        "_gate_map",
        "_gate_name_map",
        "_qarg_gate_map",
        "description",
        "_coupling_graph",
        "_unweighted_dist_matrix",
        "_length_distance_matrix",
        "_error_distance_matrix",
    )

    def __init__(self, description=None):
        """
        Create a new gate map class

        Args:
            gate_map (dict): A dictionary of gate_weight classes for keys and a list
                qargs for
            description (str): A string to describe the coupling map.
        """
        self.num_qubits = 0
        self._gate_name_map = {}
        self._gate_map = {}
        self._qarg_gate_map = {}
        self.description = description
        self._coupling_graph = None
        self._unweighted_dist_matrix = None
        self._length_distance_matrix = None
        self._error_distance_matrix = None

    def add_instruction(self, instruction, qargs, name=None, properties=None):
        """A a new gate to the gate_map

        Args:
            instruction (Instruction): The gate object to add to the map. If it's
                paramerterized any value of the parameter can be set
            qargs (list): A list of qubit indices the gate applies to. In the case
                of multiqubit gates this will be the tuple of qubits the gate can
                be applied to.
            name (str): An optional name to use for identifying the gate. If not
                specified the :attr:`~qiskit.circuit.Instruction.name` attribute
                of ``gate`` will be used. All gates in the ``Target`` need unique
                names. Using a custom name allows a backend to specify duplicate
                gates with different parameters.
            properties (dict): An optional dictionary of qarg entries to a
                InstructionProperties object for that gate implementation on the backend
        Raises:
            AttributeError: If gate is already in map
        """
        instruction_name = name or instruction.name
        if instruction_name in self._gate_map:
            raise AttributeError("already in map")
        self._gate_name_map[instruction_name] = instruction
        qargs_val = {}
        for qarg in qargs:
            self.num_qubits = max(self.num_qubits, max(qarg))
            if properties:
                if qarg in properties:
                    qargs_val[qarg] = properties[qarg]
                else:
                    qargs_val[qarg] = None
            else:
                qargs_val[qarg] = None
            if qarg in self._qarg_gate_map:
                self._qarg_gate_map[qarg].add(instruction_name)
            else:
                self._qarg_gate_map[qarg] = set(instruction_name)
        self._gate_map[instruction_name] = qargs_val
        self._coupling_graph = None
        self._unweighted_dist_matrix = None
        self._length_distance_matrix = None
        self._error_distance_matrix = None

    @property
    def qargs(self):
        """The set of qargs in the gate map."""
        return set(self._qarg_gate_map)

    def get_qargs(self, gate):
        """Get the qargs for a given gate instance

        Args:
           gate (str): The gate instance to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to
        """
        return set(self._gate_map[gate])

    def get_gate_from_name(self, gate):
        """Get the gate object for a given name

        Args:
           gate (str): The gate name to get the gate instance for
        Returns:
            Gate: The gate instance corresponding to the name
        """
        return self._gate_name_map[gate]

    def get_qargs_from_name(self, gate_name):
        """Get the qargs for a given gate by name

        Args:
           gate_name (str): The gate name to get qargs for. Note that gates
            are not necessarily unique by name (ie different parameters, user
            override, etc) this will return the qargs for any gate object that
            has a name that mateches
        Returns:
            set: The set of qargs the gate applies to.
        """
        output = set()
        for gate, qarg_map in self._gate_map.items():
            if gate == gate_name:
                output.union(set(qarg_map))
        return output

    @property
    def gate_names(self):
        """Gate the basis gate names in the gate map"""
        return set(self._gate_map)

    @property
    def gates(self):
        """Gate the basis gates in the gate map"""
        return list(self._gate_name_map.values())

    def size(self):
        """Return the number of physical qubits in this graph."""
        return self.num_qubits

    def _build_coupling_graph(self):
        self._coupling_graph = rx.PyDiGraph(multigraph=False)
        self._coupling_graph.add_nodes_from(list({} for _ in range(self.num_qubits)))
        for gate, qarg_map in self._gate_map:
            for qarg, properties in qarg_map.items():
                if len(qarg) == 1:
                    self._coupling_graph[gate] = properties
                elif len(qarg) == 2:
                    try:
                        edge_map = self._coupling_graph.has_edge(*qarg)
                        edge_map[gate] = properties
                    except rx.NoEdgeBetweenNodes:
                        self._coupling_graph.add_edge(*qarg, {gate: properties})

    def coupling_map(self):
        """Get a :class:`~qiskit.transpiler.CouplingMap` from this gate map."""
        if any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. This will not be reflected in "
                "the output coupling map."
            )
        if self._coupling_graph is None:
            self._build_coupling_graph()
        cmap = CouplingMap()
        cmap.graph = self._coupling_graph
        return cmap

    @property
    def physical_qubits(self):
        """Returns a sorted list of physical_qubits"""
        return list(range(self.num_qubits))

    def distance_matrix(self, weight=None):
        """Return the distance matrix for the coupling map."""
        if any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. These gates will not be "
                "reflected in the output matrix."
            )
        if weight is None:
            if self._unweighted_dist_matrix is None:
                self._compute_distance_matrix()
            return self._unweighted_dist_matrix
        elif weight == "error":
            if self._error_distance_matrix is None:
                self._compute_distance_matrix(weight)
            return self._error_distance_matrix
        elif weight == "length":
            if self._length_distance_matrix is None:
                self._compute_distance_matrix(weight)
            return self._length_distance_matrix
        else:
            raise TypeError("Invalid weight type %s" % weight)

    def _compute_distance_matrix(self, weight=None):
        if weight is None:
            self._unweighted_dist_matrix = rx.digraph_distance_matrix(
                self._coupling_graph, as_undirected=True
            )
        elif weight == "error":

            def error_weight_fn(edge):
                gate_props = []
                for prop in edge.values():
                    if prop is not None and prop.error is not None:
                        gate_props.append(prop.error)
                if gate_props:
                    return min(gate_props)
                else:
                    return 0

            self._error_distance_matrix = rx.digraph_floyd_warshall_numpy(
                self._coupling_graph,
                weight_fn=error_weight_fn,
                as_undirected=True,
            )
        elif weight == "length":

            def length_weight_fn(edge):
                gate_props = []
                for prop in edge.values():
                    if prop is not None and prop.length is not None:
                        gate_props.append(prop.length)
                if gate_props:
                    return min(gate_props)
                else:
                    return 0

            self._length_distance_matrix = rx.digraph_floyd_warshall_numpy(
                self._coupling_graph,
                weight_fn=length_weight_fn,
                as_undirected=True,
            )

    def distance(self, physical_qubit1, physical_qubit2, weight=None):
        """Returns the undirected distance between physical_qubit1 and physical_qubit2.

        Args:
            physical_qubit1 (int): A physical qubit
            physical_qubit2 (int): Another physical qubit
            weight (str): An optional weight function to use

        Returns:
            int: The undirected distance

        Raises:
            CouplingError: if the qubits do not exist in the CouplingMap
            TypeError: If an invalid weight is specified
        """
        if any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. These gates will not be "
                "reflected in the output matrix."
            )

        if physical_qubit1 >= self.size():
            raise CouplingError("%s not in coupling graph" % physical_qubit1)
        if physical_qubit2 >= self.size():
            raise CouplingError("%s not in coupling graph" % physical_qubit2)
        if weight is None:
            if self._unweighted_dist_matrix is None:
                self._compute_distance_matrix()
            return self._unweighted_dist_matrix[physical_qubit1, physical_qubit2]
        if weight == "error":
            if self._error_distance_matrix is None:
                self._compute_distance_matrix(weight)
            return self._error_distance_matrix[physical_qubit1, physical_qubit2]
        if weight == "length":
            if self._length_distance_matrix is None:
                self._compute_distance_matrix(weight)
            return self._error_distance_matrix[physical_qubit1, physical_qubit2]
        raise TypeError("Invalid weight %s" % weight)
