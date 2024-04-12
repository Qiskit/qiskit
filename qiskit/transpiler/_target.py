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

from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect

import rustworkx as rx

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError

# import QubitProperties here to provide convenience alias for building a
# full target
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties

# import target class from the rust side
from qiskit._accelerate.target import (
    Target as Target2,
    InstructionProperties as InstructionProperties2,
)

# TODO: Use InstructionProperties from Python side

logger = logging.getLogger(__name__)


# TODO: Leave space for InstructionProperties class
class InstructionProperties:
    """A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
    """

    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,
    ):
        self._InsrProp = InstructionProperties2(
            duration=duration, error=error, calibration=calibration
        )

    @property
    def duration(self):
        return self._InsrProp.duration

    @property
    def error(self):
        return self._InsrProp.error

    @error.setter
    def error(self, other):
        self._InsrProp.error = other

    @property
    def calibration(self):
        """The pulse representation of the instruction.

        .. note::

            This attribute always returns a Qiskit pulse program, but it is internally
            wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
            and to uniformly handle different data representation,
            for example, un-parsed Pulse Qobj JSON that a backend provider may provide.

            This value can be overridden through the property setter in following manner.
            When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
            always treated as a user-defined (custom) calibration and
            the transpiler may automatically attach the calibration data to the output circuit.
            This calibration data may appear in the wire format as an inline calibration,
            which may further update the backend standard instruction set architecture.

            If you are a backend provider who provides a default calibration data
            that is not needed to be attached to the transpiled quantum circuit,
            you can directly set :class:`.CalibrationEntry` instance to this attribute,
            in which you should set :code:`user_provided=False` when you define
            calibration data for the entry. End users can still intentionally utilize
            the calibration data, for example, to run pulse-level simulation of the circuit.
            However, such entry doesn't appear in the wire format, and backend must
            use own definition to compile the circuit down to the execution format.

        """
        return self._InsrProp.calibration

    @calibration.setter
    def calibration(self, calibration: Schedule | ScheduleBlock | CalibrationEntry):
        if isinstance(calibration, (Schedule, ScheduleBlock)):
            new_entry = ScheduleDef()
            new_entry.define(calibration, user_provided=True)
        else:
            new_entry = calibration
        self._InsrProp.calibration = new_entry

    def __repr__(self):
        return self._InsrProp.__repr__()


class Target:
    def __init__(
        self,
        description=None,
        num_qubits=0,
        dt=None,
        granularity=1,
        min_length=1,
        pulse_alignment=1,
        acquire_alignment=1,
        qubit_properties=None,
        concurrent_measurements=None,
    ):
        self._Target = Target2(
            description,
            num_qubits,
            dt,
            granularity,
            min_length,
            pulse_alignment,
            acquire_alignment,
            qubit_properties,
            concurrent_measurements,
        )

    # Convert prior attributes into properties to get dynamically
    @property
    def description(self):
        return self._Target.description

    @property
    def num_qubits(self):
        return self._Target.num_qubits

    @property
    def dt(self):
        return self._Target.dt

    @property
    def granularity(self):
        return self._Target.granularity

    @property
    def min_length(self):
        return self._Target.min_length

    @property
    def pulse_alignment(self):
        return self._Target.pulse_alignment

    @property
    def acquire_alignment(self):
        return self._Target.acquire_alignment

    @property
    def qubit_properties(self):
        return self._Target.qubit_properties

    @property
    def concurrent_measurements(self):
        return self._Target.concurrent_measurements

    @property
    def instructions(self):
        return self._Target.instructions

    def add_instruction(self, instruction, properties=None, name=None):
        """Add a new instruction to the :class:`~qiskit.transpiler.Target`

        As ``Target`` objects are strictly additive this is the primary method
        for modifying a ``Target``. Typically, you will use this to fully populate
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
            instruction (Union[qiskit.circuit.Instruction, Type[qiskit.circuit.Instruction]]):
                The operation object to add to the map. If it's parameterized any value
                of the parameter can be set. Optionally for variable width
                instructions (such as control flow operations such as :class:`~.ForLoop` or
                :class:`~MCXGate`) you can specify the class. If the class is specified than the
                ``name`` argument must be specified. When a class is used the gate is treated as global
                and not having any properties set.
            properties (dict): A dictionary of qarg entries to an
                :class:`~qiskit.transpiler.InstructionProperties` object for that
                instruction implementation on the backend. Properties are optional
                for any instruction implementation, if there are no
                :class:`~qiskit.transpiler.InstructionProperties` available for the
                backend the value can be None. If there are no constraints on the
                instruction (as in a noiseless/ideal simulation) this can be set to
                ``{None, None}`` which will indicate it runs on all qubits (or all
                available permutations of qubits for multi-qubit gates). The first
                ``None`` indicates it applies to all qubits and the second ``None``
                indicates there are no
                :class:`~qiskit.transpiler.InstructionProperties` for the
                instruction. By default, if properties is not set it is equivalent to
                passing ``{None: None}``.
            name (str): An optional name to use for identifying the instruction. If not
                specified the :attr:`~qiskit.circuit.Instruction.name` attribute
                of ``gate`` will be used. All gates in the ``Target`` need unique
                names. Backends can differentiate between different
                parameterization of a single gate by providing a unique name for
                each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
                documentation for the :class:`~qiskit.transpiler.Target` class).
        Raises:
            AttributeError: If gate is already in map
            TranspilerError: If an operation class is passed in for ``instruction`` and no name
                is specified or ``properties`` is set.
        """
        is_class = inspect.isclass(instruction)
        self._Target.add_instruction(instruction, is_class, properties, name)

    def update_instruction_properties(self, instruction, qargs, properties):
        """Update the property object for an instruction qarg pair already in the Target

        Args:
            instruction (str): The instruction name to update
            qargs (tuple): The qargs to update the properties of
            properties (InstructionProperties): The properties to set for this instruction
        Raises:
            KeyError: If ``instruction`` or ``qarg`` are not in the target
        """
        self._Target.update_instruction_properties(instruction, qargs, properties)

    def update_from_instruction_schedule_map(self, inst_map, inst_name_map=None, error_dict=None):
        """Update the target from an instruction schedule map.

        If the input instruction schedule map contains new instructions not in
        the target they will be added. However, if it contains additional qargs
        for an existing instruction in the target it will error.

        Args:
            inst_map (InstructionScheduleMap): The instruction
            inst_name_map (dict): An optional dictionary that maps any
                instruction name in ``inst_map`` to an instruction object.
                If not provided, instruction is pulled from the standard Qiskit gates,
                and finally custom gate instance is created with schedule name.
            error_dict (dict): A dictionary of errors of the form::

                {gate_name: {qarg: error}}

            for example::

                {'rx': {(0, ): 1.4e-4, (1, ): 1.2e-4}}

            For each entry in the ``inst_map`` if ``error_dict`` is defined
            a when updating the ``Target`` the error value will be pulled from
            this dictionary. If one is not found in ``error_dict`` then
            ``None`` will be used.
        """
        get_calibration = getattr(inst_map, "_get_calibration_entry")
        # Expand name mapping with custom gate name provided by user.
        qiskit_inst_name_map = get_standard_gate_name_mapping()
        if inst_name_map is not None:
            qiskit_inst_name_map.update(inst_name_map)

        for inst_name in inst_map.instructions:
            # Prepare dictionary of instruction properties
            out_props = {}
            for qargs in inst_map.qubits_with_instruction(inst_name):
                try:
                    qargs = tuple(qargs)
                except TypeError:
                    qargs = (qargs,)
                try:
                    props = self._Target.gate_map[inst_name][qargs]
                except (KeyError, TypeError):
                    props = None

                entry = get_calibration(inst_name, qargs)
                if entry.user_provided and getattr(props, "calibration", None) != entry:
                    # It only copies user-provided calibration from the inst map.
                    # Backend defined entry must already exist in Target.
                    if self.dt is not None:
                        try:
                            duration = entry.get_schedule().duration * self.dt
                        except UnassignedDurationError:
                            # duration of schedule is parameterized
                            duration = None
                    else:
                        duration = None
                    props = InstructionProperties(
                        duration=duration,
                        calibration=entry,
                    )
                else:
                    if props is None:
                        # Edge case. Calibration is backend defined, but this is not
                        # registered in the backend target. Ignore this entry.
                        continue
                try:
                    # Update gate error if provided.
                    props.error = error_dict[inst_name][qargs]
                except (KeyError, TypeError):
                    pass
                out_props[qargs] = props
            if not out_props:
                continue
            # Prepare Qiskit Gate object assigned to the entries
            if inst_name not in self._Target.gate_map:
                # Entry not found: Add new instruction
                if inst_name in qiskit_inst_name_map:
                    # Remove qargs with length that doesn't match with instruction qubit number
                    inst_obj = qiskit_inst_name_map[inst_name]
                    normalized_props = {}
                    for qargs, prop in out_props.items():
                        if len(qargs) != inst_obj.num_qubits:
                            continue
                        normalized_props[qargs] = prop
                    self.add_instruction(inst_obj, normalized_props, name=inst_name)
                else:
                    # Check qubit length parameter name uniformity.
                    qlen = set()
                    param_names = set()
                    for qargs in inst_map.qubits_with_instruction(inst_name):
                        if isinstance(qargs, int):
                            qargs = (qargs,)
                        qlen.add(len(qargs))
                        cal = getattr(out_props[tuple(qargs)], "calibration")
                        param_names.add(tuple(cal.get_signature().parameters.keys()))
                    if len(qlen) > 1 or len(param_names) > 1:
                        raise QiskitError(
                            f"Schedules for {inst_name} are defined non-uniformly for "
                            f"multiple qubit lengths {qlen}, "
                            f"or different parameter names {param_names}. "
                            "Provide these schedules with inst_name_map or define them with "
                            "different names for different gate parameters."
                        )
                    inst_obj = Gate(
                        name=inst_name,
                        num_qubits=next(iter(qlen)),
                        params=list(map(Parameter, next(iter(param_names)))),
                    )
                    self.add_instruction(inst_obj, out_props, name=inst_name)
            else:
                # Entry found: Update "existing" instructions.
                for qargs, prop in out_props.items():
                    if qargs not in self._Target.gate_map[inst_name]:
                        continue
                    self.update_instruction_properties(inst_name, qargs, prop)

    def instruction_schedule_map(self):
        """Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions in the target with a pulse schedule defined.

        Returns:
            InstructionScheduleMap: The instruction schedule map for the
            instructions in this target with a pulse schedule defined.
        """
        out_inst_schedule_map = InstructionScheduleMap()
        return self._Target.instruction_schedule_map(out_inst_schedule_map)

    @property
    def qargs(self):
        return self._Target.qargs

    def qargs_for_operation_name(self, operation):
        """Get the qargs for a given operation name

        Args:
           operation (str): The operation name to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to.
        """
        return {x: None for x in self._Target.qargs_for_operation_name(operation)}.keys()

    def durations(self):
        """Get an InstructionDurations object from the target

        Returns:
            InstructionDurations: The instruction duration represented in the
                target
        """
        if self._Target.instruction_durations is not None:
            return self._instruction_durations
        out_durations = []
        for instruction, props_map in self._Target.gate_map.items():
            for qarg, properties in props_map.items():
                if properties is not None and properties.duration is not None:
                    out_durations.append((instruction, list(qarg), properties.duration, "s"))
        self._Target.instruction_durations = InstructionDurations(out_durations, dt=self.dt)
        return self._Target.instruction_durations

    def timing_constraints(self):
        """Get an :class:`~qiskit.transpiler.TimingConstraints` object from the target

        Returns:
            TimingConstraints: The timing constraints represented in the ``Target``
        """
        return TimingConstraints(
            self.granularity, self.min_length, self.pulse_alignment, self.acquire_alignment
        )

    def operation_from_name(self, instruction):
        """Get the operation class object for a given name

        Args:
            instruction (str): The instruction name to get the
                :class:`~qiskit.circuit.Instruction` instance for
        Returns:
            qiskit.circuit.Instruction: The Instruction instance corresponding to the
            name. This also can also be the class for globally defined variable with
            operations.
        """
        return self._Target.gate_name_map[instruction]
    
    def operation_names_for_qargs(self, qargs):
        """Get the operation names for a specified qargs tuple

        Args:
            qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return the names for any globally defined operations in the target.
        Returns:
            set: The set of operation names that apply to the specified ``qargs``.

        Raises:
            KeyError: If ``qargs`` is not in target
        """
        self._Target.operation_names_for_qargs(inspect.isclass, qargs)
    
    