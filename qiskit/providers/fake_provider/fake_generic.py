# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generic fake BackendV2 class"""

from __future__ import annotations
import warnings

from collections.abc import Iterable
import numpy as np

from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import Measure, Parameter, Delay, Reset, QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    BreakLoopOp,
    ContinueLoopOp,
)
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
    PulseDefaults,
    Command,
)
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals


class GenericTarget(Target):
    """
    This class will generate a :class:`~.Target` instance with
    default qubit, instruction and calibration properties.
    A target object represents the minimum set of information
    the transpiler needs from a backend.
    """

    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        control_flow: bool = False,
        calibrate_instructions: list[str] | None = None,
        rng: np.random.Generator = None,
    ):
        """
        Args:
            num_qubits (int): Number of qubits in the target.

            basis_gates (list[str]): List of basis gate names to be supported by
                the target. These must be within the supported operations of this
                target generator, which can be consulted through the
                ``supported_operations`` property.
                Common sets of basis gates are ``{"cx", "id", "rz", "sx", "x"}``
                and ``{"ecr", "id", "rz", "sx", "x"}``. The ``"reset"``,  ``"`delay"``
                and ``"measure"`` instructions are supported by default, even if
                not specified via ``basis_gates``.

            coupling_map (CouplingMap): Target's coupling map as an instance of
                :class:`~.CouplingMap`.

            control_flow (bool): Flag to enable control flow directives on the target
                (defaults to False).

            calibrate_instructions (list[str] | None): List of instruction names
                to add default calibration entries to. These must be within the
                supported operations of this target generator, which can be
                consulted through the ``supported_operations`` property. If no
                instructions are provided, the target generator will append
                empty calibration schedules by default.

            rng (np.random.Generator): Optional fixed-seed generator for default random values.
        """
        self._rng = rng if rng else np.random.default_rng(seed=42)

        if num_qubits != coupling_map.size():
            raise QiskitError(
                f"The number of qubits (got {num_qubits}) must match "
                f"the size of the provided coupling map (got {coupling_map.size()})."
            )

        self._num_qubits = num_qubits
        self._coupling_map = coupling_map

        # Hardcoded default target attributes. To modify,
        # access corresponding properties through the public `Target` API
        super().__init__(
            description="Generic Target",
            num_qubits=num_qubits,
            dt=0.222e-9,
            qubit_properties=[
                QubitProperties(
                    t1=self._rng.uniform(100e-6, 200e-6),
                    t2=self._rng.uniform(100e-6, 200e-6),
                    frequency=self._rng.uniform(5e9, 5.5e9),
                )
                for _ in range(num_qubits)
            ],
            concurrent_measurements=[list(range(num_qubits))],
        )

        # Ensure that Reset, Delay and Measure are in
        # self._basis_gates
        # so that their instructions are added to the target.
        self._basis_gates = basis_gates
        for name in ["reset", "delay", "measure"]:
            if name not in self._basis_gates:
                self._basis_gates.append(name)

        # Iterate over gates, generate noise params from defaults,
        # and add instructions to target.
        for name in self._basis_gates:
            if name not in self.supported_operations:
                raise QiskitError(
                    f"Provided base gate {name} is not a supported "
                    f"operation ({self.supported_operations.keys()})."
                )
            gate = self.supported_operations[name]
            noise_params = self.noise_defaults[name]
            self.add_noisy_instruction(gate, noise_params)

        if control_flow:
            self.add_instruction(IfElseOp, name="if_else")
            self.add_instruction(WhileLoopOp, name="while_loop")
            self.add_instruction(ForLoopOp, name="for_loop")
            self.add_instruction(SwitchCaseOp, name="switch_case")
            self.add_instruction(BreakLoopOp, name="break")
            self.add_instruction(ContinueLoopOp, name="continue")

        # Generate block of calibration defaults and add to target.
        # Note: this could be improved if we could generate and add
        # calibration defaults per-gate, and not as a block.
        defaults = self._generate_calibration_defaults(calibrate_instructions)
        inst_map = defaults.instruction_schedule_map
        self.add_calibrations_from_instruction_schedule_map(inst_map)

    @property
    def supported_operations(self) -> dict[str, Instruction]:
        """Mapping of names to class instances for operations supported
        in ``basis_gates``.

        Returns:
            Dictionary mapping operation names to class instances.
        """
        return {
            "cx": CXGate(),
            "ecr": ECRGate(),
            "id": IGate(),
            "rz": RZGate(Parameter("theta")),
            "sx": SXGate(),
            "x": XGate(),
            "measure": Measure(),
            "delay": Delay(Parameter("Time")),
            "reset": Reset(),
        }

    @property
    def noise_defaults(self) -> dict[str, tuple | None]:
        """Noise default values/ranges for duration and error of supported
         instructions. There are three possible formats:

            #. (min_duration, max_duration, min_error, max_error),
                if the defaults are ranges
            #. (duration, error), if the defaults are fixed values
            #. None

        Returns:
            Dictionary mapping instruction names to noise defaults
        """
        return {
            "cx": (1e-5, 5e-3, 1e-8, 9e-7),
            "ecr": (1e-5, 5e-3, 1e-8, 9e-7),
            "id": (0.0, 0.0),
            "rz": (0.0, 0.0),
            "sx": (1e-5, 5e-3, 1e-8, 9e-7),
            "x": (1e-5, 5e-3, 1e-8, 9e-7),
            "measure": (1e-5, 5e-3, 1e-8, 9e-7),
            "delay": None,
            "reset": None,
        }

    def add_noisy_instruction(
        self, instruction: Instruction, noise_params: tuple[float, ...] | None
    ) -> None:
        """Add instruction properties to target for specified instruction.

        Args:
            instruction (Instruction): Instance of instruction to be added to the target
            noise_params (tuple[float, ...] | None): error and duration noise values/ranges to
                include in instruction properties.

        Returns:
            None
        """

        qarg_set = self._coupling_map if instruction.num_qubits > 1 else range(self._num_qubits)
        props = {}

        for qarg in qarg_set:
            try:
                qargs = tuple(qarg)
            except TypeError:
                qargs = (qarg,)

            duration, error = (
                (None, None)
                if noise_params is None
                else noise_params
                if len(noise_params) == 2
                else (self._rng.uniform(*noise_params[:2]), self._rng.uniform(*noise_params[2:]))
            )

            props.update({qargs: InstructionProperties(duration, error)})

        self.add_instruction(instruction, props)

    def add_calibrations_from_instruction_schedule_map(
        self, inst_map: InstructionScheduleMap
    ) -> None:
        """Add calibration entries from provided pulse defaults to target.

        Args:
            inst_map (InstructionScheduleMap): pulse defaults with instruction schedule map

        Returns:
            None
        """

        # The calibration entries are directly injected into the gate map to
        # avoid then being labeled as "user_provided".
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qobj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in self._gate_map:
                    if inst == "measure":
                        for qubit in qargs:
                            self._gate_map[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in self._gate_map[inst] and inst not in ["delay", "reset"]:
                        self._gate_map[inst][qargs].calibration = calibration_entry

    def _generate_calibration_defaults(
        self, calibrate_instructions: list[str] | None
    ) -> PulseDefaults:
        """Generate calibration defaults for instructions specified via ``calibrate_instructions``.
        By default, this method generates empty calibration schedules.

        Args:
            calibrate_instructions (list[str]): list of instructions to be calibrated.

        Returns:
            Corresponding PulseDefaults
        """

        # The number of samples determines the pulse durations of the corresponding
        # instructions. This class generates pulses with durations in multiples of
        # 16 for consistency with the pulse granularity of real IBM devices, but
        # keeps the number smaller than what would be realistic for
        # manageability. If needed, more realistic durations could be added in the
        # future (order of 160 dt for 1q gates, 1760 for 2q gates and measure).

        samples_1 = np.linspace(0, 1.0, 16, dtype=np.complex128)  # 16dt
        samples_2 = np.linspace(0, 1.0, 32, dtype=np.complex128)  # 32dt
        samples_3 = np.linspace(0, 1.0, 64, dtype=np.complex128)  # 64dt

        pulse_library = [
            PulseLibraryItem(name="pulse_1", samples=samples_1),
            PulseLibraryItem(name="pulse_2", samples=samples_2),
            PulseLibraryItem(name="pulse_3", samples=samples_3),
        ]

        # Unless explicitly given a series of gates to calibrate, this method
        # will generate empty pulse schedules for all gates in self._basis_gates.
        calibrate_instructions = calibrate_instructions or []
        calibration_buffer = self._basis_gates.copy()
        for inst in ["delay", "reset"]:
            calibration_buffer.remove(inst)

        # List of calibration commands (generated from sequences of PulseQobjInstructions)
        # corresponding to each calibrated instruction. Note that the calibration pulses
        # are different for 1q gates vs 2q gates vs measurement instructions.
        cmd_def = []
        for inst in calibration_buffer:
            num_qubits = self.supported_operations[inst].num_qubits
            qarg_set = self._coupling_map if num_qubits > 1 else list(range(self._num_qubits))
            if inst == "measure":
                sequence = []
                qubits = qarg_set
                if inst in calibrate_instructions:
                    sequence = [
                        PulseQobjInstruction(
                            name="acquire",
                            duration=1792,
                            t0=0,
                            qubits=list(range(self.num_qubits)),
                            memory_slot=list(range(self.num_qubits)),
                        )
                    ] + [PulseQobjInstruction(name="pulse_2", ch=f"m{i}", t0=0) for i in qarg_set]
                cmd_def.append(
                    Command(
                        name=inst,
                        qubits=qubits,
                        sequence=sequence,
                    )
                )
            else:
                for qarg in qarg_set:
                    sequence = []
                    qubits = [qarg] if num_qubits == 1 else qarg
                    if inst in calibrate_instructions:
                        if num_qubits == 1:
                            sequence = [
                                PulseQobjInstruction(name="fc", ch=f"u{qarg}", t0=0, phase="-P0"),
                                PulseQobjInstruction(name="pulse_1", ch=f"d{qarg}", t0=0),
                            ]
                        else:
                            sequence = [
                                PulseQobjInstruction(name="pulse_2", ch=f"d{qarg[0]}", t0=0),
                                PulseQobjInstruction(name="pulse_3", ch=f"u{qarg[0]}", t0=0),
                                PulseQobjInstruction(name="pulse_2", ch=f"d{qarg[1]}", t0=0),
                                PulseQobjInstruction(name="fc", ch=f"d{qarg[1]}", t0=0, phase=2.1),
                            ]
                    cmd_def.append(
                        Command(
                            name=inst,
                            qubits=qubits,
                            sequence=sequence,
                        )
                    )

        qubit_freq_est = np.random.normal(4.8, scale=0.01, size=self.num_qubits).tolist()
        meas_freq_est = np.linspace(6.4, 6.6, self.num_qubits).tolist()
        return PulseDefaults(
            qubit_freq_est=qubit_freq_est,
            meas_freq_est=meas_freq_est,
            buffer=0,
            pulse_library=pulse_library,
            cmd_def=cmd_def,
        )


class FakeGeneric(BackendV2):
    """
    Configurable fake :class:`~.BackendV2` generator. This class will
    generate a fake backend from a combination of generated defaults
    (with a fixable ``seed``) driven from a series of optional input arguments.
    """

    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str],
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_instructions: list[str] | None = None,
        seed: int = 42,
    ):
        """
        Args:
           num_qubits (int): Number of qubits that will
                be used to construct the backend's target. Note that, while
                there is no limit in the size of the target that can be
                constructed, fake backends run on local noisy simulators,
                and these might show limitations in the number of qubits that
                can be simulated.

            basis_gates (list[str]): List of basis gate names to be supported by
                the target. These must be within the supported operations of the
                target generator, which can be consulted through its
                ``supported_operations`` property.
                Common sets of basis gates are ``{"cx", "id", "rz", "sx", "x"}``
                and ``{"ecr", "id", "rz", "sx", "x"}``. The ``"reset"``,  ``"`delay"``,
                and ``"measure"`` instructions are supported by default, even if
                not specified via ``basis_gates``.

            coupling_map (list[list[int]] | CouplingMap | None): Optional coupling map
                for the fake backend. Multiple formats are supported:

                #. :class:`~.CouplingMap` instance
                #. List, must be given as an adjacency matrix, where each entry
                   specifies all directed two-qubit interactions supported by the backend,
                   e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

                If ``coupling_map`` is specified, it must match the number of qubits
                specified in ``num_qubits``. If ``coupling_map`` is not specified,
                a fully connected coupling map will be generated with ``num_qubits``
                qubits.

            control_flow (bool): Flag to enable control flow directives on the target
                (defaults to False).

            calibrate_instructions (list[str] | None): List of instruction names
                to add default calibration entries to. These must be within the
                supported operations of the target generator, which can be
                consulted through the ``supported_operations`` property. If no
                instructions are provided, the target generator will append
                empty calibration schedules by default.

            seed (int): Optional seed for generation of default values.
        """

        super().__init__(
            provider=None,
            name="fake_generic",
            description=f"This is a fake device with {num_qubits} " f"and generic settings.",
            backend_version="",
        )
        self._rng = np.random.default_rng(seed=seed)

        # the coupling map is necessary to build the default channels
        if coupling_map is None:
            self._coupling_map = CouplingMap().from_full(num_qubits)
        else:
            if isinstance(coupling_map, CouplingMap):
                self._coupling_map = coupling_map
            else:
                self._coupling_map = CouplingMap(coupling_map)

        self._target = GenericTarget(
            num_qubits,
            basis_gates,
            self._coupling_map,
            control_flow,
            calibrate_instructions,
            self._rng,
        )

        self._build_default_channels()
        self.sim = None

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def meas_map(self) -> list[list[int]]:
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int):
        drive_channels_map = getattr(self, "channels_map", {}).get("drive", {})
        qubits = (qubit,)
        if qubits in drive_channels_map:
            return drive_channels_map[qubits][0]
        return None

    def measure_channel(self, qubit: int):
        measure_channels_map = getattr(self, "channels_map", {}).get("measure", {})
        qubits = (qubit,)
        if qubits in measure_channels_map:
            return measure_channels_map[qubits][0]
        return None

    def acquire_channel(self, qubit: int):
        acquire_channels_map = getattr(self, "channels_map", {}).get("acquire", {})
        qubits = (qubit,)
        if qubits in acquire_channels_map:
            return acquire_channels_map[qubits][0]
        return None

    def control_channel(self, qubits: Iterable[int]):
        control_channels_map = getattr(self, "channels_map", {}).get("control", {})
        qubits = tuple(qubits)
        if qubits in control_channels_map:
            return control_channels_map[qubits]
        return []

    def run(self, run_input, **options):
        """Run on the fake backend using a simulator.

        This method runs circuit jobs (an individual or a list of :class:`~.QuantumCircuit`
        ) and pulse jobs (an individual or a list of :class:`~.Schedule` or
        :class:`~.ScheduleBlock`) using :class:`~.BasicAer` or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using the ``AerSimulator`` with
        noise model of the fake backend. Otherwise, jobs will be run using the
        ``BasicAer`` simulator without noise.

        Noisy simulations of pulse jobs are not yet supported in :class:`~.FakeGeneric`.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object, then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit_aer is not installed.
        """

        circuits = run_input
        pulse_job = None
        if isinstance(circuits, (pulse.Schedule, pulse.ScheduleBlock)):
            pulse_job = True
        elif isinstance(circuits, QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all(isinstance(x, (pulse.Schedule, pulse.ScheduleBlock)) for x in circuits):
                    pulse_job = True
                elif all(isinstance(x, QuantumCircuit) for x in circuits):
                    pulse_job = False
        if pulse_job is None:  # submitted job is invalid
            raise QiskitError(
                "Invalid input object %s, must be either a "
                "QuantumCircuit, Schedule, or a list of either" % circuits
            )
        if pulse_job:  # pulse job
            raise QiskitError("Pulse simulation is currently not supported for V2 fake backends.")
        # circuit job
        if not _optionals.HAS_AER:
            warnings.warn("Aer not found using BasicAer and no noise", RuntimeWarning)
        if self.sim is None:
            self._setup_sim()
        self.sim._options = self._options
        job = self.sim.run(circuits, **options)
        return job

    def _setup_sim(self) -> None:

        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            self.sim = AerSimulator()
            noise_model = NoiseModel.from_backend(self)
            self.sim.set_options(noise_model=noise_model)
            # Update fake backend default too to avoid overwriting
            # it when run() is called
            self.set_options(noise_model=noise_model)

        else:
            self.sim = BasicAer.get_backend("qasm_simulator")

    @classmethod
    def _default_options(cls) -> Options:

        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator

            return AerSimulator._default_options()
        else:
            return BasicAer.get_backend("qasm_simulator")._default_options()

    def _build_default_channels(self) -> None:

        channels_map = {
            "acquire": {(i,): [pulse.AcquireChannel(i)] for i in range(self.num_qubits)},
            "drive": {(i,): [pulse.DriveChannel(i)] for i in range(self.num_qubits)},
            "measure": {(i,): [pulse.MeasureChannel(i)] for i in range(self.num_qubits)},
            "control": {
                (edge): [pulse.ControlChannel(i)] for i, edge in enumerate(self._coupling_map)
            },
        }
        setattr(self, "channels_map", channels_map)
