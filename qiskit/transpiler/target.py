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

from collections.abc import Mapping
import logging
from typing import Union, Dict, Any

import retworkx as rx

from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.transpiler.instruction_durations import InstructionDurations

logger = logging.getLogger(__name__)


class InstructionProperties:
    """A representation of the properties of a gate implementation."""

    __slots__ = ("length", "error", "pulse", "properties")

    def __init__(
        self,
        length: float = None,
        error: float = None,
        pulse: Union["Schedule", "ScheduleBlock"] = None,
        properties: Dict[str, Any] = None,
    ):
        """Create a new ``InstructionProperties`` object

        Args:
            length: The duration of the instruction on the specified set of
                qubits
            error: The average error rate for the instruction on the specified
                set of qubits.
            pulse: The pulse representation of the instruction
            properties: A free form dictionary of additional properties the
                backend has for a specified instruction (operation + arguments).
        """
        self.length = length
        self.error = error
        self.pulse = pulse
        self.properties = properties

    def __repr__(self):
        return (
            f"InstructionProperties(length={self.length}, error={self.error}"
            f", pulse={self.pulse}, properties={self.properties})"
        )


class Target(Mapping):
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
        gmap.add_instruction(UGate(theta, phi, lam), [(0,), (1,)], properties=u_props)
        cx_props = {
            (0,1): InstructionProperties(length=5.23e-7, error=0.00098115),
            (1,0): InstructionProperties(length=4.52e-7, error=0.00132115),
        }
        gmap.add_instruction(CXGate(), [(0, 1), (1, 0)], properties=cx_props)

    The intent of the ``Target`` object is to inform Qiskit's compiler about
    the constraints of a particular backend so the compiler can compile an
    input circuit to something that works and is optimized for a device.

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
        "_instruction_durations",
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
        # A mapping of gate name -> gate instance
        self._gate_name_map = {}
        # A nested mapping of gate name -> qargs -> properties
        self._gate_map = {}
        # A mapping of qarg -> set(gate name)
        self._qarg_gate_map = {}
        self.description = description
        self._coupling_graph = None
        self._unweighted_dist_matrix = None
        self._length_distance_matrix = None
        self._error_distance_matrix = None
        self._instruction_durations = None

    def add_instruction(self, instruction, properties, name=None):
        """Add a new instruction to the :class:`~qiskit.transpiler.Target`

        As ``Target`` objects are strictly additive this is the primary method
        for modifying a ``Target``. Typically you will use this to fully populate
        a ``Target`` before using it in :class:`~qiskit.providers.BackendV2`. For
        example::

            from qiskit.circuit.library import CXGate
            from qiskit.transpiler import Target, InstructionProperties

            target = Target()
            cx_properties = {
                (0, 1): None,
                (1, 0): None,
                (0, 2): None,
                (2, 0): None,
                (0, 3): None,
                (2, 3): None,
                (3, 0): None,
                (3, 2): None
            }
            target.add_instruction(CXGate(), cx_properties)

        Will add a :class:`~qiskit.circuit.library.CXGate` to the target with no
        properties (duration, error, etc) with the coupling edge list:
        ``(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (2, 3), (3, 0), (3, 2)``. If
        there are properties available for the instruction you can replace the
        ``None`` value in the properties dictionary with an
        :class:`~qiskit.transpiler.InstructionProperties` object. This pattern
        is repeated for each :class:`~qiskit.circuit.Instruction` the target
        supports.

        Args:
            instruction (Instruction): The gate object to add to the map. If it's
                paramerterized any value of the parameter can be set
            properties (dict): A dictionary of qarg entries to an
                :class:`~qiskit.transpiler.InstructionProperties` object for that
                instruction implementation on the backend. Properties are optional
                for any instruction implementation, if there are no
                :class:`~qiskit.transpiler.InstructionProperties` available for the
                backend the value can be None.
            name (str): An optional name to use for identifying the gate. If not
                specified the :attr:`~qiskit.circuit.Instruction.name` attribute
                of ``gate`` will be used. All gates in the ``Target`` need unique
                names. Using a custom name allows a backend to specify duplicate
                gates with different parameters.
        Raises:
            AttributeError: If gate is already in map
        """
        instruction_name = name or instruction.name
        if instruction_name in self._gate_map:
            raise AttributeError("Instruction %s is already in the target" % instruction_name)
        self._gate_name_map[instruction_name] = instruction
        qargs_val = {}
        for qarg in properties:
            self.num_qubits = max(self.num_qubits, max(qarg) + 1)
            qargs_val[qarg] = properties[qarg]
            if qarg in self._qarg_gate_map:
                self._qarg_gate_map[qarg].add(instruction_name)
            else:
                self._qarg_gate_map[qarg] = {
                    instruction_name,
                }
        self._gate_map[instruction_name] = qargs_val
        self._coupling_graph = None
        self._unweighted_dist_matrix = None
        self._length_distance_matrix = None
        self._error_distance_matrix = None
        self._instruction_durations = None

    @property
    def qargs(self):
        """The set of qargs in the gate map."""
        return set(self._qarg_gate_map)

    def get_qargs(self, gate):
        """Get the qargs for a given gate name

        Args:
           gate (str): The gate instance to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to
        """
        return set(self._gate_map[gate])

    def durations(self):
        """Get an InstructionDurations object from the target

        Returns:
            InstructionDurations: The instruction duration represented in the
                target
        """
        if self._instruction_durations is not None:
            return self._instruction_durations
        out_durations = []
        for instruction, props_map in self._gate_map.items():
            for qarg, properties in props_map.items():
                if properties is not None and properties.length is not None:
                    out_durations.append((instruction, list(qarg), properties.length, "s"))
        self._instruction_durations = InstructionDurations(out_durations)
        return self._instruction_durations

    def get_gate_from_name(self, gate):
        """Get the gate object for a given name

        Args:
           gate (str): The gate name to get the gate instance for
        Returns:
            Gate: The gate instance corresponding to the name
        """
        return self._gate_name_map[gate]

    def get_instructions_for_qargs(self, qarg):
        """Get the qargs for a given gate by name

        Args:
            qarg (tuple): A qarg tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0.
        Returns:
            list: The set of :class:`~qiskit.circuit.Instruction` instances
            that apply to the specified qarg.
        """
        return [self._gate_name_map[x] for x in self._qarg_gate_map[qarg]]

    @property
    def instruction_names(self):
        """Gate the basis instruction names in the gate map"""
        return set(self._gate_map)

    @property
    def instructions(self):
        """Gate the instruction gates in the gate map"""
        return list(self._gate_name_map.values())

    def _build_coupling_graph(self):
        self._coupling_graph = rx.PyDiGraph(multigraph=False)
        self._coupling_graph.add_nodes_from(list({} for _ in range(self.num_qubits)))
        for gate, qarg_map in self._gate_map.items():
            for qarg, properties in qarg_map.items():
                if len(qarg) == 1:
                    self._coupling_graph[qarg[0]] = properties
                elif len(qarg) == 2:
                    try:
                        edge_data = self._coupling_graph.get_edge_data(*qarg)
                        edge_data[gate] = properties
                    except rx.NoEdgeBetweenNodes:
                        self._coupling_graph.add_edge(*qarg, {gate: properties})

    def coupling_map(self, two_q_gate=None):
        """Get a :class:`~qiskit.transpiler.CouplingMap` from this gate map.

        Args:
            two_q_gate (str): An optional gate name for a two qubit gate in
                the Target to generate the coupling map for. If specified the
                output coupling map will only have edges between qubits where
                this gate is present.
        Returns:
            CouplingMap: The :class:`~qiskit.transpiler.CouplingMap` object
                for this target.

        Raises:
            ValueError: If a non-two qubit gate is passed in for ``two_q_gate``.
            IndexError: If an Instruction not in the Target is passed in for
                ``two_q_gate``.
        """
        if any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. This will not be reflected in "
                "the output coupling map."
            )

        if two_q_gate is not None:
            coupling_graph = rx.PyDiGraph(multigraph=False)
            coupling_graph.add_nodes_from(list(None for _ in range(self.num_qubits)))
            for qargs, properties in self._gate_map[two_q_gate].items():
                if len(qargs) != 2:
                    raise ValueError(
                        "Specified two_q_gate: %s is not a 2 qubit instruction" % two_q_gate
                    )
                coupling_graph.add_edge(*qargs, {two_q_gate: properties})
            cmap = CouplingMap()
            cmap.graph = coupling_graph
            return cmap

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
        if self._coupling_graph is None:
            self._build_coupling_graph()
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

        if physical_qubit1 >= self.num_qubits:
            raise CouplingError("%s not in coupling graph" % physical_qubit1)
        if physical_qubit2 >= self.num_qubits:
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

    def __iter__(self):
        return iter(self._gate_map)

    def __getitem__(self, key):
        return self._gate_map[key]

    def __len__(self):
        return len(self._gate_map)

    def __contains__(self, item):
        return item in self._gate_map

    def keys(self):
        return self._gate_map.keys()

    def values(self):
        return self._gate_map.values()

    def items(self):
        return self._gate_map.items()
