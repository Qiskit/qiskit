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

# pylint: disable=too-many-return-statements

"""
A target object represents the minimum set of information the transpiler needs
from a backend
"""

from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect

import retworkx as rx

from qiskit.circuit.parameter import Parameter
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints

# import QubitProperties here to provide convenience alias for building a
# full target
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties

logger = logging.getLogger(__name__)


class InstructionProperties:
    """A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
    """

    __slots__ = ("duration", "error", "calibration")

    def __init__(
        self,
        duration: float = None,
        error: float = None,
        calibration=None,
    ):
        """Create a new ``InstructionProperties`` object

        Args:
            duration: The duration, in seconds, of the instruction on the
                specified set of qubits
            error: The average error rate for the instruction on the specified
                set of qubits.
            calibration (Union["qiskit.pulse.Schedule", "qiskit.pulse.ScheduleBlock"]): The pulse
                representation of the instruction
        """
        self.duration = duration
        self.error = error
        self.calibration = calibration

    def __repr__(self):
        return (
            f"InstructionProperties(duration={self.duration}, error={self.error}"
            f", calibration={self.calibration})"
        )


class Target(Mapping):
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

    Each instruction in the Target is indexed by a unique string name that uniquely
    identifies that instance of an :class:`~qiskit.circuit.Instruction` object in
    the Target. There is a 1:1 mapping between a name and an
    :class:`~qiskit.circuit.Instruction` instance in the target and each name must
    be unique. By default the name is the :attr:`~qiskit.circuit.Instruction.name`
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
        would potentially be invalidated by removals.
    """

    __slots__ = (
        "num_qubits",
        "_gate_map",
        "_gate_name_map",
        "_qarg_gate_map",
        "description",
        "_coupling_graph",
        "_instruction_durations",
        "_instruction_schedule_map",
        "dt",
        "granularity",
        "min_length",
        "pulse_alignment",
        "aquire_alignment",
        "_non_global_basis",
        "_non_global_strict_basis",
        "qubit_properties",
        "_global_operations",
    )

    def __init__(
        self,
        description=None,
        num_qubits=0,
        dt=None,
        granularity=1,
        min_length=1,
        pulse_alignment=1,
        aquire_alignment=1,
        qubit_properties=None,
    ):
        """
        Create a new Target object

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
        Raises:
            ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
            defined and the value of ``num_qubits`` differs from the length of
            ``qubit_properties``.
        """
        self.num_qubits = num_qubits
        # A mapping of gate name -> gate instance
        self._gate_name_map = {}
        # A nested mapping of gate name -> qargs -> properties
        self._gate_map = {}
        # A mapping of number of qubits to set of op names which are global
        self._global_operations = defaultdict(set)
        # A mapping of qarg -> set(gate name)
        self._qarg_gate_map = defaultdict(set)
        self.dt = dt
        self.description = description
        self._coupling_graph = None
        self._instruction_durations = None
        self._instruction_schedule_map = None
        self.granularity = granularity
        self.min_length = min_length
        self.pulse_alignment = pulse_alignment
        self.aquire_alignment = aquire_alignment
        self._non_global_basis = None
        self._non_global_strict_basis = None
        if qubit_properties is not None:
            if not self.num_qubits:
                self.num_qubits = len(qubit_properties)
            else:
                if self.num_qubits != len(qubit_properties):
                    raise ValueError(
                        "The value of num_qubits specified does not match the "
                        "length of the input qubit_properties list"
                    )
        self.qubit_properties = qubit_properties

    def add_instruction(self, instruction, properties=None, name=None):
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
            instruction (qiskit.circuit.Instruction): The operation object to add to the map. If it's
                paramerterized any value of the parameter can be set. Optionally for variable width
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
                instruction (as in a noisless/ideal simulation) this can be set to
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
                parameterizations of a single gate by providing a unique name for
                each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
                documentation for the :class:`~qiskit.transpiler.Target` class).
        Raises:
            AttributeError: If gate is already in map
            TranspilerError: If an operation class is passed in for ``instruction`` and no name
                is specified or ``properties`` is set.
        """
        is_class = inspect.isclass(instruction)
        if not is_class:
            instruction_name = name or instruction.name
        else:
            # Invalid to have class input without a name with characters set "" is not a valid name
            if not name:
                raise TranspilerError(
                    "A name must be specified when defining a supported global operation by class"
                )
            if properties is not None:
                raise TranspilerError(
                    "An instruction added globally by class can't have properties set."
                )
            instruction_name = name
        if properties is None:
            properties = {None: None}
        if instruction_name in self._gate_map:
            raise AttributeError("Instruction %s is already in the target" % instruction_name)
        self._gate_name_map[instruction_name] = instruction
        if is_class:
            qargs_val = {None: None}
        else:
            if None in properties:
                self._global_operations[instruction.num_qubits].add(instruction_name)
            qargs_val = {}
            for qarg in properties:
                if qarg is not None and len(qarg) != instruction.num_qubits:
                    raise TranspilerError(
                        f"The number of qubits for {instruction} does not match the number "
                        f"of qubits in the properties dictionary: {qarg}"
                    )
                if qarg is not None:
                    self.num_qubits = max(self.num_qubits, max(qarg) + 1)
                qargs_val[qarg] = properties[qarg]
                self._qarg_gate_map[qarg].add(instruction_name)
        self._gate_map[instruction_name] = qargs_val
        self._coupling_graph = None
        self._instruction_durations = None
        self._instruction_schedule_map = None
        self._non_global_basis = None
        self._non_global_strict_basis = None

    def update_instruction_properties(self, instruction, qargs, properties):
        """Update the property object for an instruction qarg pair already in the Target

        Args:
            instruction (str): The instruction name to update
            qargs (tuple): The qargs to update the properties of
            properties (InstructionProperties): The properties to set for this nstruction
        Raises:
            KeyError: If ``instruction`` or ``qarg`` are not in the target
        """
        if instruction not in self._gate_map:
            raise KeyError(f"Provided instruction: '{instruction}' not in this Target")
        if qargs not in self._gate_map[instruction]:
            raise KeyError(f"Provided qarg: '{qargs}' not in this Target for {instruction}")
        self._gate_map[instruction][qargs] = properties
        self._instruction_durations = None
        self._instruction_schedule_map = None

    def update_from_instruction_schedule_map(self, inst_map, inst_name_map=None, error_dict=None):
        """Update the target from an instruction schedule map.

        If the input instruction schedule map contains new instructions not in
        the target they will be added. However if it contains additional qargs
        for an existing instruction in the target it will error.

        Args:
            inst_map (InstructionScheduleMap): The instruction
            inst_name_map (dict): An optional dictionary that maps any
                instruction name in ``inst_map`` to an instruction object
            error_dict (dict): A dictionary of errors of the form::

                {gate_name: {qarg: error}}

            for example::

                {'rx': {(0, ): 1.4e-4, (1, ): 1.2e-4}}

            For each entry in the ``inst_map`` if ``error_dict`` is defined
            a when updating the ``Target`` the error value will be pulled from
            this dictionary. If one is not found in ``error_dict`` then
            ``None`` will be used.

        Raises:
            ValueError: If ``inst_map`` contains new instructions and
                ``inst_name_map`` isn't specified
            KeyError: If a ``inst_map`` contains a qarg for an instruction
                that's not in the target
        """
        for inst in inst_map.instructions:
            out_props = {}
            for qarg in inst_map.qubits_with_instruction(inst):
                sched = inst_map.get(inst, qarg)
                val = InstructionProperties(calibration=sched)
                try:
                    qarg = tuple(qarg)
                except TypeError:
                    qarg = (qarg,)
                if inst in self._gate_map:
                    if self.dt is not None:
                        val.duration = sched.duration * self.dt
                    else:
                        val.duration = None
                    if error_dict is not None:
                        error_inst = error_dict.get(inst)
                        if error_inst:
                            error = error_inst.get(qarg)
                            val.error = error
                        else:
                            val.error = None
                    else:
                        val.error = None
                out_props[qarg] = val
            if inst not in self._gate_map:
                if inst_name_map is not None:
                    self.add_instruction(inst_name_map[inst], out_props, name=inst)
                else:
                    raise ValueError(
                        "An inst_name_map kwarg must be specified to add new "
                        "instructions from an InstructionScheduleMap"
                    )
            else:
                for qarg, prop in out_props.items():
                    self.update_instruction_properties(inst, qarg, prop)

    @property
    def qargs(self):
        """The set of qargs in the target."""
        qargs = set(self._qarg_gate_map)
        if len(qargs) == 1 and next(iter(qargs)) is None:
            return None
        return qargs

    def qargs_for_operation_name(self, operation):
        """Get the qargs for a given operation name

        Args:
           operation (str): The operation name to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to.
        """
        if None in self._gate_map[operation]:
            return None
        return self._gate_map[operation].keys()

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
                if properties is not None and properties.duration is not None:
                    out_durations.append((instruction, list(qarg), properties.duration, "s"))
        self._instruction_durations = InstructionDurations(out_durations, dt=self.dt)
        return self._instruction_durations

    def timing_constraints(self):
        """Get an :class:`~qiskit.transpiler.TimingConstraints` object from the target

        Returns:
            TimingConstraints: The timing constraints represented in the Target
        """
        return TimingConstraints(
            self.granularity, self.min_length, self.pulse_alignment, self.aquire_alignment
        )

    def instruction_schedule_map(self):
        """Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions in the target with a pulse schedule defined.

        Returns:
            InstructionScheduleMap: The instruction schedule map for the
            instructions in this target with a pulse schedule defined.
        """
        if self._instruction_schedule_map is not None:
            return self._instruction_schedule_map
        out_inst_schedule_map = InstructionScheduleMap()
        for instruction, qargs in self._gate_map.items():
            for qarg, properties in qargs.items():
                if properties is not None and properties.calibration is not None:
                    out_inst_schedule_map.add(instruction, qarg, properties.calibration)
        self._instruction_schedule_map = out_inst_schedule_map
        return out_inst_schedule_map

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
        return self._gate_name_map[instruction]

    def operations_for_qargs(self, qargs):
        """Get the operation class object for a specified qargs tuple

        Args:
            qargs (tuple): A qargs tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return any globally defined operations in the target.
        Returns:
            list: The list of :class:`~qiskit.circuit.Instruction` instances
            that apply to the specified qarg. This may also be a class if
            a variable width operation is globally defined.

        Raises:
            KeyError: If qargs is not in target
        """
        if qargs is not None and any(x not in range(0, self.num_qubits) for x in qargs):
            raise KeyError(f"{qargs} not in target.")
        res = [self._gate_name_map[x] for x in self._qarg_gate_map[qargs]]
        if qargs is not None:
            res += self._global_operations.get(len(qargs), list())
        for op in self._gate_name_map.values():
            if inspect.isclass(op):
                res.append(op)
        if not res:
            raise KeyError(f"{qargs} not in target.")
        return list(res)

    def operation_names_for_qargs(self, qargs):
        """Get the operation names for a specified qargs tuple

        Args:
            qargs (tuple): A qargs tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return the names for any globally defined operations in the target.
        Returns:
            set: The set of operation names that apply to the specified
            `qargs``.

        Raises:
            KeyError: If qargs is not in target
        """
        if qargs is not None and any(x not in range(0, self.num_qubits) for x in qargs):
            raise KeyError(f"{qargs} not in target.")
        res = self._qarg_gate_map.get(qargs, set())
        if qargs is not None:
            res.update(self._global_operations.get(len(qargs), set()))
        for name, op in self._gate_name_map.items():
            if inspect.isclass(op):
                res.add(name)
        if not res:
            raise KeyError(f"{qargs} not in target.")
        return res

    def instruction_supported(
        self, operation_name=None, qargs=None, operation_class=None, parameters=None
    ):
        """Return whether the instruction (operation + qubits) is supported by the target

        Args:
            operation_name (str): The name of the operation for the instruction. Either
                this or ``operation_class`` must be specified, if both are specified
                ``operation_class`` will take priority and this argument will be ignored.
            qargs (tuple): The tuple of qubit indices for the instruction. If this is
                not specified then this method will return ``True`` if the specified
                operation is supported on any qubits. The typical application will
                always have this set (otherwise it's the same as just checking if the
                target contains the operation). Normally you would not set this argument
                if you wanted to check more generally that the target supports an operation
                with the ``parameters`` on any qubits.
            operation_class (qiskit.circuit.Instruction): The operation class to check whether
                the target supports a particular operation by class rather
                than by name. This lookup is more expensive as it needs to
                iterate over all operations in the target instead of just a
                single lookup. If this is specified it will supersede the
                ``operation_name`` argument. The typical use case for this
                operation is to check whether a specific variant of an operation
                is supported on the backend. For example, if you wanted to
                check whether a :class:`~.RXGate` was supported on a specific
                qubit with a fixed angle. That fixed angle variant will
                typically have a name different than the object's
                :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
                This can be used to check if any instances of the class are
                available in such a case.
            parameters (list): A list of parameters to check if the target
                supports them on the specified qubits. If the instruction
                supports the parameter values specified in the list on the
                operation and qargs specified this will return ``True`` but
                if the parameters are not supported on the specified
                instruction it will return ``False``. If this argument is not
                specified this method will return ``True`` if the instruction
                is supported independent of the instruction parameters. If
                specified with any :class:`~.Parameter` objects in the list,
                that entry will be treated as supporting any value, however parameter names
                will not be checked (for example if an operation in the target
                is listed as parameterized with ``"theta"`` and ``"phi"`` is
                passed into this function that will return ``True``). For
                example, if called with::

                    parameters = [Parameter("theta")]
                    target.instruction_supported("rx", (0,), parameters=parameters)

                will return ``True`` if an :class:`~.RXGate` is suporrted on qubit 0
                that will accept any parameter. If you need to check for a fixed numeric
                value parameter this argument is typically paired with the ``operation_class``
                argument. For example::

                    target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])

                will return ``True`` if an RXGate(pi/4) exists on qubit 0.

        Returns:
            bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.

        """

        def check_obj_params(parameters, obj):
            for index, param in enumerate(parameters):
                if isinstance(param, Parameter) and not isinstance(obj.params[index], Parameter):
                    return False
                if param != obj.params[index] and not isinstance(obj.params[index], Parameter):
                    return False
            return True

        # Case a list if passed in by mistake
        if qargs is not None:
            qargs = tuple(qargs)
        if operation_class is not None:
            for op_name, obj in self._gate_name_map.items():
                if inspect.isclass(obj):
                    if obj != operation_class:
                        continue
                    # If no qargs a operation class is supported
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False

                if isinstance(obj, operation_class):
                    if parameters is not None:
                        if len(parameters) != len(obj.params):
                            continue
                        if not check_obj_params(parameters, obj):
                            continue
                    if qargs is None:
                        return True
                    if qargs in self._gate_map[op_name]:
                        return True
                    if self._gate_map[op_name] is None or None in self._gate_map[op_name]:
                        return self._gate_name_map[op_name].num_qubits == len(qargs) and all(
                            x < self.num_qubits for x in qargs
                        )
            return False
        if operation_name in self._gate_map:
            if parameters is not None:
                obj = self._gate_name_map[operation_name]
                if inspect.isclass(obj):
                    # The parameters argument was set and the operation_name specified is
                    # defined as a globally supported class in the target. This means
                    # there is no available validation (including whether the specified
                    # operation supports parameters), the returned value will not factor
                    # in the argument `parameters`,

                    # If no qargs a operation class is supported
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False
                if len(parameters) != len(obj.params):
                    return False
                for index, param in enumerate(parameters):
                    matching_param = False
                    if isinstance(obj.params[index], Parameter):
                        matching_param = True
                    elif param == obj.params[index]:
                        matching_param = True
                    if not matching_param:
                        return False
                return True
            if qargs is None:
                return True
            if qargs in self._gate_map[operation_name]:
                return True
            if self._gate_map[operation_name] is None or None in self._gate_map[operation_name]:
                obj = self._gate_name_map[operation_name]
                if inspect.isclass(obj):
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False
                else:
                    return self._gate_name_map[operation_name].num_qubits == len(qargs) and all(
                        x < self.num_qubits for x in qargs
                    )
        return False

    @property
    def operation_names(self):
        """Get the operation names in the target."""
        return self._gate_map.keys()

    @property
    def operations(self):
        """Get the operation class objects in the target."""
        return list(self._gate_name_map.values())

    @property
    def instructions(self):
        """Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
        for the target

        For globally defined variable width operations the tuple will be of the form
        ``(class, None)`` where class is the actual operation class that
        is globally defined.
        """
        return [
            (self._gate_name_map[op], qarg) for op in self._gate_map for qarg in self._gate_map[op]
        ]

    def instruction_properties(self, index):
        """Get the instruction properties for a specific instruction tuple

        This method is to be used in conjunction with the
        :attr:`~qiskit.transpiler.Target.instructions` attribute of a
        :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
        get the instruction properties for an element of
        :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
        However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
        directly it is likely more efficient to access the target directly via the name
        and qubits to get the instruction properties. For example, if
        :attr:`~qiskit.transpiler.Target.instructions` returned::

            [(XGate(), (0,)), (XGate(), (1,))]

        you could get the properties of the ``XGate`` on qubit 1 with::

            props = target.instruction_properties(1)

        but just accessing it directly via the name would be more efficient::

            props = target['x'][(1,)]

        (assuming the ``XGate``'s canonical name in the target is ``'x'``)
        This is especially true for larger targets as this will scale worse with the number
        of instruction tuples in a target.

        Args:
            index (int): The index of the instruction tuple from the
                :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
                if you want the properties from the third element in
                :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
        Returns:
            InstructionProperties: The instruction properties for the specified instruction tuple
        """
        instruction_properties = [
            inst_props for op in self._gate_map for _, inst_props in self._gate_map[op].items()
        ]
        return instruction_properties[index]

    def _build_coupling_graph(self):
        self._coupling_graph = rx.PyDiGraph(multigraph=False)
        self._coupling_graph.add_nodes_from([{} for _ in range(self.num_qubits)])
        for gate, qarg_map in self._gate_map.items():
            if qarg_map is None:
                if self._gate_name_map[gate].num_qubits == 2:
                    self._coupling_graph = None
                    return
                continue
            for qarg, properties in qarg_map.items():
                if qarg is None:
                    if self._gate_name_map[gate].num_qubits == 2:
                        self._coupling_graph = None
                        return
                    continue
                if len(qarg) == 1:
                    self._coupling_graph[qarg[0]] = properties
                elif len(qarg) == 2:
                    try:
                        edge_data = self._coupling_graph.get_edge_data(*qarg)
                        edge_data[gate] = properties
                    except rx.NoEdgeBetweenNodes:
                        self._coupling_graph.add_edge(*qarg, {gate: properties})
        if self._coupling_graph.num_edges() == 0 and any(x is None for x in self._qarg_gate_map):
            self._coupling_graph = None

    def build_coupling_map(self, two_q_gate=None):
        """Get a :class:`~qiskit.transpiler.CouplingMap` from this target.

        If there is a mix of two qubit operations that have a connectivity
        constraint and those that are globally defined this will also return
        ``None`` because the globally connectivity means there is no contstraint
        on the target. If you wish to see the constraints of the two qubit
        operations that have constraints you should use the ``two_q_gate``
        argument to limit the output to the gates which have a constraint.

        Args:
            two_q_gate (str): An optional gate name for a two qubit gate in
                the Target to generate the coupling map for. If specified the
                output coupling map will only have edges between qubits where
                this gate is present.
        Returns:
            CouplingMap: The :class:`~qiskit.transpiler.CouplingMap` object
                for this target. If there are no connectivity constraints in
                the target this will return ``None``.

        Raises:
            ValueError: If a non-two qubit gate is passed in for ``two_q_gate``.
            IndexError: If an Instruction not in the Target is passed in for
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
        # if there is no connectivity constraints in the coupling graph treat it as not
        # existing and return
        if self._coupling_graph is not None:
            cmap = CouplingMap()
            cmap.graph = self._coupling_graph
            return cmap
        else:
            return None

    @property
    def physical_qubits(self):
        """Returns a sorted list of physical_qubits"""
        return list(range(self.num_qubits))

    def get_non_global_operation_names(self, strict_direction=False):
        """Return the non-global operation names for the target

        The non-global operations are those in the target which don't apply
        on all qubits (for single qubit operations) or all multiqubit qargs
        (for multi-qubit operations).

        Args:
            strict_direction (bool): If set to ``True`` the multi-qubit
                operations considered as non-global respect the strict
                direction (or order of qubits in the qargs is signifcant). For
                example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
                defined over ``(1, 0)`` by default neither would be considered
                non-global, but if ``strict_direction`` is set ``True`` both
                ``cx`` and ``ecr`` would be returned.

        Returns:
            List[str]: A list of operation names for operations that aren't global in this target
        """
        if strict_direction:
            if self._non_global_strict_basis is not None:
                return self._non_global_strict_basis
            search_set = self._qarg_gate_map.keys()
        else:
            if self._non_global_basis is not None:
                return self._non_global_basis

            search_set = {
                frozenset(qarg)
                for qarg in self._qarg_gate_map
                if qarg is not None and len(qarg) != 1
            }
        incomplete_basis_gates = []
        size_dict = defaultdict(int)
        size_dict[1] = self.num_qubits
        for qarg in search_set:
            if qarg is None or len(qarg) == 1:
                continue
            size_dict[len(qarg)] += 1
        for inst, qargs in self._gate_map.items():
            qarg_sample = next(iter(qargs))
            if qarg_sample is None:
                continue
            if not strict_direction:
                qargs = {frozenset(qarg) for qarg in qargs}
            if len(qargs) != size_dict[len(qarg_sample)]:
                incomplete_basis_gates.append(inst)
        if strict_direction:
            self._non_global_strict_basis = incomplete_basis_gates
        else:
            self._non_global_basis = incomplete_basis_gates
        return incomplete_basis_gates

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

    def __str__(self):
        output = io.StringIO()
        if self.description is not None:
            output.write(f"Target: {self.description}\n")
        else:
            output.write("Target\n")
        output.write(f"Number of qubits: {self.num_qubits}\n")
        output.write("Instructions:\n")
        for inst, qarg_props in self._gate_map.items():
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
                    prop_str_pieces.append(f"\t\t\tDuration: {duration} sec.\n")
                error = getattr(props, "error", None)
                if error is not None:
                    prop_str_pieces.append(f"\t\t\tError Rate: {error}\n")
                schedule = getattr(props, "calibration", None)
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

    properties_dict = {
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
                if props is not None:
                    if props.duration is not None:
                        property_list.append(
                            {
                                "date": datetime.datetime.utcnow(),
                                "name": "gate_length",
                                "unit": "s",
                                "value": props.duration,
                            }
                        )
                    if props.error is not None:
                        property_list.append(
                            {
                                "date": datetime.datetime.utcnow(),
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
            qubit_props = {x: None for x in range(target.num_qubits)}
            for qargs, props in qargs_list.items():
                if qargs is None:
                    continue
                qubit = qargs[0]
                props_list = []
                if props is not None:
                    if props.error is not None:
                        props_list.append(
                            {
                                "date": datetime.datetime.utcnow(),
                                "name": "readout_error",
                                "unit": "",
                                "value": props.error,
                            }
                        )
                    if props.duration is not None:
                        props_list.append(
                            {
                                "date": datetime.datetime.utcnow(),
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
