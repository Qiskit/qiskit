# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-return-statements

"""
A target object represents the minimum set of information the transpiler needs
from a backend
"""

from __future__ import annotations

import itertools

from typing import Any
import datetime
import io
import logging

import rustworkx as rx

from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.instruction_durations import (  # pylint: disable=unused-import
    InstructionDurations,
)
from qiskit.transpiler.timing_constraints import TimingConstraints  # pylint: disable=unused-import

# import QubitProperties here to provide convenience alias for building a
# full target
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock

# import target class from the rust side
from qiskit._accelerate.target import (  # pylint: disable=unused-import
    Target as Target2,
    InstructionProperties as InstructionProperties2,
)

logger = logging.getLogger(__name__)


class InstructionProperties(InstructionProperties2):
    """A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
    """

    def __new__(
        cls,
        duration=None,
        error=None,
        calibration=None,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        return InstructionProperties2.__new__(
            cls, duration=duration, error=error, calibration=calibration
        )

    def __init__(
        self,
        duration=None,  # pylint: disable=unused-argument
        error=None,  # pylint: disable=unused-argument
        calibration=None,  # pylint: disable=unused-argument
    ):
        """Create a new ``InstructionProperties`` object

        Args:
            duration: The duration, in seconds, of the instruction on the
                specified set of qubits
            error: The average error rate for the instruction on the specified
                set of qubits.
            calibration: The pulse representation of the instruction.
        """

    def __repr__(self):
        return (
            f"InstructionProperties(duration={self.duration}, error={self.error}"
            f", calibration={self._calibration})"
        )


class Target(Target2):
    """
        The intent of the ``Target`` object is to inform Qiskit's compiler about
    the constraints of a particular backend so the compiler can compile an
    input circuit to something that works and is optimized for a device. It
    currently contains a description of instructions on a backend and their
    properties as well as some timing information. However, this exact
    interface may evolve over time as the needs of the compiler change. These
    changes will be done in a backwards compatible and controlled manner when
    they are made (either through versioning, subclassing, or mixins) to add
    on to the set of information exposed by a target.

    As a basic example, let's assume backend has two qubits, supports
    :class:`~qiskit.circuit.library.UGate` on both qubits and
    :class:`~qiskit.circuit.library.CXGate` in both directions. To model this
    you would create the target like::

        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit.library import UGate, CXGate
        from qiskit.circuit import Parameter

        gmap = Target()
        theta = Parameter('theta')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        u_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(duration=4.52e-8, error=0.00032115),
        }
        gmap.add_instruction(UGate(theta, phi, lam), u_props)
        cx_props = {
            (0,1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1,0): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }
        gmap.add_instruction(CXGate(), cx_props)

    Each instruction in the ``Target`` is indexed by a unique string name that uniquely
    identifies that instance of an :class:`~qiskit.circuit.Instruction` object in
    the Target. There is a 1:1 mapping between a name and an
    :class:`~qiskit.circuit.Instruction` instance in the target and each name must
    be unique. By default, the name is the :attr:`~qiskit.circuit.Instruction.name`
    attribute of the instruction, but can be set to anything. This lets a single
    target have multiple instances of the same instruction class with different
    parameters. For example, if a backend target has two instances of an
    :class:`~qiskit.circuit.library.RXGate` one is parameterized over any theta
    while the other is tuned up for a theta of pi/6 you can add these by doing something
    like::

        import math

        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit.library import RXGate
        from qiskit.circuit import Parameter

        target = Target()
        theta = Parameter('theta')
        rx_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
        }
        target.add_instruction(RXGate(theta), rx_props)
        rx_30_props = {
            (0,): InstructionProperties(duration=1.74e-6, error=.00012)
        }
        target.add_instruction(RXGate(math.pi / 6), rx_30_props, name='rx_30')

    Then in the ``target`` object accessing by ``rx_30`` will get the fixed
    angle :class:`~qiskit.circuit.library.RXGate` while ``rx`` will get the
    parameterized :class:`~qiskit.circuit.library.RXGate`.

    .. note::

        This class assumes that qubit indices start at 0 and are a contiguous
        set if you want a submapping the bits will need to be reindexed in
        a new``Target`` object.

    .. note::

        This class only supports additions of gates, qargs, and qubits.
        If you need to remove one of these the best option is to iterate over
        an existing object and create a new subset (or use one of the methods
        to do this). The object internally caches different views and these
        would potentially be invalidated by removals."""

    def __new__(
        cls,
        description: str | None = None,
        num_qubits: int = 0,
        dt: float | None = None,
        granularity: int = 1,
        min_length: int = 1,
        pulse_alignment: int = 1,
        acquire_alignment: int = 1,
        qubit_properties: list | None = None,
        concurrent_measurements: list | None = None,
    ):
        """
        Create a new ``Target`` object

        Args:
            description (str): An optional string to describe the Target.
            num_qubits (int): An optional int to specify the number of qubits
                the backend target has. If not set it will be implicitly set
                based on the qargs when :meth:`~qiskit.Target.add_instruction`
                is called. Note this must be set if the backend target is for a
                noiseless simulator that doesn't have constraints on the
                instructions so the transpiler knows how many qubits are
                available.
            dt (float): The system time resolution of input signals in seconds
            granularity (int): An integer value representing minimum pulse gate
                resolution in units of ``dt``. A user-defined pulse gate should
                have duration of a multiple of this granularity value.
            min_length (int): An integer value representing minimum pulse gate
                length in units of ``dt``. A user-defined pulse gate should be
                longer than this length.
            pulse_alignment (int): An integer value representing a time
                resolution of gate instruction starting time. Gate instruction
                should start at time which is a multiple of the alignment
                value.
            acquire_alignment (int): An integer value representing a time
                resolution of measure instruction starting time. Measure
                instruction should start at time which is a multiple of the
                alignment value.
            qubit_properties (list): A list of :class:`~.QubitProperties`
                objects defining the characteristics of each qubit on the
                target device. If specified the length of this list must match
                the number of qubits in the target, where the index in the list
                matches the qubit number the properties are defined for. If some
                qubits don't have properties available you can set that entry to
                ``None``
            concurrent_measurements(list): A list of sets of qubits that must be
                measured together. This must be provided
                as a nested list like ``[[0, 1], [2, 3, 4]]``.
        Raises:
            ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
                defined and the value of ``num_qubits`` differs from the length of
                ``qubit_properties``.
        """

        # In case a number is passed as first argument, assume it means num_qubits.
        if description is not None:
            if num_qubits is None and isinstance(description, int):
                num_qubits = description
                description = None
            elif not isinstance(description, str):
                description = str(description)

        return super(Target, cls).__new__(
            cls,
            description=description,
            num_qubits=num_qubits,
            dt=dt,
            granularity=granularity,
            min_length=min_length,
            pulse_alignment=pulse_alignment,
            acquire_alignment=acquire_alignment,
            qubit_properties=qubit_properties,
            concurrent_measurements=concurrent_measurements,
        )

    @property
    def operation_names(self):
        """Get the operation names in the target."""
        return {x: None for x in super().operation_names}.keys()

    def _build_coupling_graph(self):
        self.coupling_graph = rx.PyDiGraph(  # pylint: disable=attribute-defined-outside-init
            multigraph=False
        )
        self.coupling_graph.add_nodes_from([{} for _ in range(self.num_qubits)])
        for gate, qarg_map in self.items():
            if qarg_map is None:
                if self.gate_name_map[gate].num_qubits == 2:
                    self.coupling_graph = None  # pylint: disable=attribute-defined-outside-init
                    return
                continue
            for qarg, properties in qarg_map.items():
                if qarg is None:
                    if self.operation_from_name(gate).num_qubits == 2:
                        self.coupling_graph = None  # pylint: disable=attribute-defined-outside-init
                        return
                    continue
                if len(qarg) == 1:
                    self.coupling_graph[qarg[0]] = (
                        properties  # pylint: disable=attribute-defined-outside-init
                    )
                elif len(qarg) == 2:
                    try:
                        edge_data = self.coupling_graph.get_edge_data(*qarg)
                        edge_data[gate] = properties
                    except rx.NoEdgeBetweenNodes:
                        self.coupling_graph.add_edge(*qarg, {gate: properties})
        qargs = self.qargs
        if self.coupling_graph.num_edges() == 0 and (
            qargs == None or any(x is None for x in qargs)
        ):
            self.coupling_graph = None  # pylint: disable=attribute-defined-outside-init

    def build_coupling_map(self, two_q_gate=None, filter_idle_qubits=False):
        """Get a :class:`~qiskit.transpiler.CouplingMap` from this target.

        If there is a mix of two qubit operations that have a connectivity
        constraint and those that are globally defined this will also return
        ``None`` because the globally connectivity means there is no constraint
        on the target. If you wish to see the constraints of the two qubit
        operations that have constraints you should use the ``two_q_gate``
        argument to limit the output to the gates which have a constraint.

        Args:
            two_q_gate (str): An optional gate name for a two qubit gate in
                the ``Target`` to generate the coupling map for. If specified the
                output coupling map will only have edges between qubits where
                this gate is present.
            filter_idle_qubits (bool): If set to ``True`` the output :class:`~.CouplingMap`
                will remove any qubits that don't have any operations defined in the
                target. Note that using this argument will result in an output
                :class:`~.CouplingMap` object which has holes in its indices
                which might differ from the assumptions of the class. The typical use
                case of this argument is to be paired with
                :meth:`.CouplingMap.connected_components` which will handle the holes
                as expected.
        Returns:
            CouplingMap: The :class:`~qiskit.transpiler.CouplingMap` object
                for this target. If there are no connectivity constraints in
                the target this will return ``None``.

        Raises:
            ValueError: If a non-two qubit gate is passed in for ``two_q_gate``.
            IndexError: If an Instruction not in the ``Target`` is passed in for
                ``two_q_gate``.
        """
        if self.qargs is None:
            return None
        if None not in self.qargs and any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. This will not be reflected in "
                "the output coupling map."
            )

        if two_q_gate is not None:
            coupling_graph = rx.PyDiGraph(multigraph=False)
            coupling_graph.add_nodes_from([None] * self.num_qubits)
            for qargs, properties in self[two_q_gate].items():
                if len(qargs) != 2:
                    raise ValueError(
                        "Specified two_q_gate: %s is not a 2 qubit instruction" % two_q_gate
                    )
                coupling_graph.add_edge(*qargs, {two_q_gate: properties})
            cmap = CouplingMap()
            cmap.graph = coupling_graph
            return cmap
        if self.coupling_graph is None:
            self._build_coupling_graph()
        # if there is no connectivity constraints in the coupling graph treat it as not
        # existing and return
        if self.coupling_graph is not None:
            cmap = CouplingMap()
            if filter_idle_qubits:
                cmap.graph = self._filter_coupling_graph()
            else:
                cmap.graph = self.coupling_graph.copy()
            return cmap
        else:
            return None

    def _filter_coupling_graph(self):
        has_operations = set(itertools.chain.from_iterable(x for x in self.qargs if x is not None))
        graph = self.coupling_graph.copy()
        to_remove = set(graph.node_indices()).difference(has_operations)
        if to_remove:
            graph.remove_nodes_from(list(to_remove))
        return graph

    # Magic methods

    def __iter__(self):
        """Returns an iterator over the names of each supported gate"""
        return iter(super().__iter__())

    def keys(self):
        """Return all gate names present in the Target"""
        return {x: None for x in super().keys()}.keys()

    def __str__(self):
        output = io.StringIO()
        if self.description is not None:
            output.write(f"Target: {self.description}\n")
        else:
            output.write("Target\n")
        output.write(f"Number of qubits: {self.num_qubits}\n")
        output.write("Instructions:\n")
        for inst, qarg_props in super().items():
            output.write(f"\t{inst}\n")
            for qarg, props in qarg_props.items():
                if qarg is None:
                    continue
                if props is None:
                    output.write(f"\t\t{qarg}\n")
                    continue
                prop_str_pieces = [f"\t\t{qarg}:\n"]
                duration = getattr(props, "duration", None)
                if duration is not None:
                    prop_str_pieces.append(
                        f"\t\t\tDuration: {0 if duration == 0 else duration} sec.\n"
                    )
                error = getattr(props, "error", None)
                if error is not None:
                    prop_str_pieces.append(f"\t\t\tError Rate: {0 if error == 0 else error}\n")
                schedule = getattr(props, "_calibration", None)
                if schedule is not None:
                    prop_str_pieces.append("\t\t\tWith pulse schedule calibration\n")
                extra_props = getattr(props, "properties", None)
                if extra_props is not None:
                    extra_props_pieces = [
                        f"\t\t\t\t{key}: {value}\n" for key, value in extra_props.items()
                    ]
                    extra_props_str = "".join(extra_props_pieces)
                    prop_str_pieces.append(f"\t\t\tExtra properties:\n{extra_props_str}\n")
                output.write("".join(prop_str_pieces))
        return output.getvalue()


def target_to_backend_properties(target: Target):
    """Convert a :class:`~.Target` object into a legacy :class:`~.BackendProperties`"""

    properties_dict: dict[str, Any] = {
        "backend_name": "",
        "backend_version": "",
        "last_update_date": None,
        "general": [],
    }
    gates = []
    qubits = []
    for gate, qargs_list in target.items():
        if gate != "measure":
            for qargs, props in qargs_list.items():
                property_list = []
                if getattr(props, "duration", None) is not None:
                    property_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "gate_length",
                            "unit": "s",
                            "value": props.duration,
                        }
                    )
                if getattr(props, "error", None) is not None:
                    property_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "gate_error",
                            "unit": "",
                            "value": props.error,
                        }
                    )
                if property_list:
                    gates.append(
                        {
                            "gate": gate,
                            "qubits": list(qargs),
                            "parameters": property_list,
                            "name": gate + "_".join([str(x) for x in qargs]),
                        }
                    )
        else:
            qubit_props: dict[int, Any] = {}
            if target.num_qubits is not None:
                qubit_props = {x: None for x in range(target.num_qubits)}
            for qargs, props in qargs_list.items():
                if qargs is None:
                    continue
                qubit = qargs[0]
                props_list = []
                if getattr(props, "error", None) is not None:
                    props_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "readout_error",
                            "unit": "",
                            "value": props.error,
                        }
                    )
                if getattr(props, "duration", None) is not None:
                    props_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "readout_length",
                            "unit": "s",
                            "value": props.duration,
                        }
                    )
                if not props_list:
                    qubit_props = {}
                    break
                qubit_props[qubit] = props_list
            if qubit_props and all(x is not None for x in qubit_props.values()):
                qubits = [qubit_props[i] for i in range(target.num_qubits)]
    if gates or qubits:
        properties_dict["gates"] = gates
        properties_dict["qubits"] = qubits
        return BackendProperties.from_dict(properties_dict)
    else:
        return None
