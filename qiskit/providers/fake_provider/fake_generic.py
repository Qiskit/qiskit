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
    """

    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str],
        coupling_map: CouplingMap,
        control_flow: bool = False,
        calibrate_gates: list[str] | None = None,
        rng: np.random.Generator = None,
    ):
        """
        Args:
            num_qubits (int): Number of qubits.

            basis_gates (list[str]): List of basis gate names to be supported by
                the target. The currently supported instructions can be consulted via
                the ``supported_instructions`` property.
                Common sets of basis gates are ``{"cx", "id", "rz", "sx", "x"}``
                and ``{"ecr", "id", "rz", "sx", "x"}``.

            coupling_map (CouplingMap): Target's coupling map as an instance of
                :class:`~.CouplingMap`.

            control_flow (bool): Flag to enable control flow directives on the backend
                (defaults to False).

            calibrate_gates (list[str] | None): List of gate names which should contain
                default calibration entries (overriden if an ``instruction_schedule_map`` is
                provided). These must be a subset of ``basis_gates``.

            rng (np.random.Generator): Optional fixed-seed generator for default random values.
        """
        self._rng = rng if rng else np.random.default_rng(seed=42)
        self._num_qubits = num_qubits
        self._coupling_map = coupling_map

        # hardcode default target attributes. To modify,
        # access corresponding properties through the public Target API
        super().__init__(
            description="Fake Target",
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

        # ensure that reset, delay and measure are in basis_gates
        self._basis_gates = set(basis_gates)
        for name in {"reset", "delay", "measure"}:
            self._basis_gates.add(name)

        # iterate over gates, generate noise params from defaults
        # and add instructions to target
        for name in self._basis_gates:
            if name not in self.supported_instructions:
                raise QiskitError(
                    f"Provided base gate {name} is not a supported "
                    f"instruction ({self.supported_instructions})."
                )
            gate = self.supported_instructions[name]
            noise_params = self.noise_defaults[name]
            self.add_noisy_instruction(gate, noise_params)

        if control_flow:
            self.add_instruction(IfElseOp, name="if_else")
            self.add_instruction(WhileLoopOp, name="while_loop")
            self.add_instruction(ForLoopOp, name="for_loop")
            self.add_instruction(SwitchCaseOp, name="switch_case")
            self.add_instruction(BreakLoopOp, name="break")
            self.add_instruction(ContinueLoopOp, name="continue")

        # generate block of calibration defaults and add to target
        if calibrate_gates is not None:
            defaults = self._generate_calibration_defaults(calibrate_gates)
            self.add_calibration_defaults(defaults)

    @property
    def supported_instructions(self) -> dict[str, Instruction]:
        """Mapping of names to class instances for instructions supported
        in ``basis_gates``.

        Returns:
            Dictionary mapping instruction names to instruction instances
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

    def add_calibration_defaults(self, defaults: PulseDefaults) -> None:
        """Add calibration entries from provided pulse defaults to target.

        Args:
            defaults (PulseDefaults): pulse defaults with instruction schedule map

        Returns:
            None
        """
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qpbj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in self._gate_map:
                    if inst == "measure":
                        for qubit in qargs:
                            self._gate_map[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in self._gate_map[inst] and inst not in ["delay", "reset"]:
                        self._gate_map[inst][qargs].calibration = calibration_entry

    def _generate_calibration_defaults(self, calibrate_gates: list[str] | None) -> PulseDefaults:
        """Generate calibration defaults for specified gates.

        Args:
            calibrate_gates (list[str]): list of gates to be calibrated.

        Returns:
            Corresponding PulseDefaults
        """
        measure_command_sequence = [
            PulseQobjInstruction(
                name="acquire",
                duration=1792,
                t0=0,
                qubits=list(range(self.num_qubits)),
                memory_slot=list(range(self.num_qubits)),
            )
        ]

        measure_command_sequence += [
            PulseQobjInstruction(name="pulse_1", ch=f"m{i}", duration=1792, t0=0)
            for i in range(self.num_qubits)
        ]

        measure_command = Command(
            name="measure",
            qubits=list(range(self.num_qubits)),
            sequence=measure_command_sequence,
        )

        cmd_def = [measure_command]

        for gate in self._basis_gates:
            for i in range(self.num_qubits):
                sequence = []
                if gate in calibrate_gates:
                    sequence = [
                        PulseQobjInstruction(name="fc", ch=f"d{i}", t0=0, phase="-P0"),
                        PulseQobjInstruction(name="pulse_3", ch=f"d{i}", t0=0),
                    ]
                cmd_def.append(
                    Command(
                        name=gate,
                        qubits=[i],
                        sequence=sequence,
                    )
                )

        for qubit1, qubit2 in self._coupling_map:
            sequence = [
                PulseQobjInstruction(name="pulse_1", ch=f"d{qubit1}", t0=0),
                PulseQobjInstruction(name="pulse_2", ch=f"u{qubit1}", t0=10),
                PulseQobjInstruction(name="pulse_1", ch=f"d{qubit2}", t0=20),
                PulseQobjInstruction(name="fc", ch=f"d{qubit2}", t0=20, phase=2.1),
            ]

            if "cx" in self._basis_gates:
                if "cx" in calibrate_gates:
                    sequence = []
                cmd_def += [
                    Command(
                        name="cx",
                        qubits=[qubit1, qubit2],
                        sequence=sequence,
                    )
                ]
            if "ecr" in self._basis_gates:
                if "ecr" in calibrate_gates:
                    sequence = []
                cmd_def += [
                    Command(
                        name="ecr",
                        qubits=[qubit1, qubit2],
                        sequence=sequence,
                    )
                ]

        qubit_freq_est = np.random.normal(4.8, scale=0.01, size=self.num_qubits).tolist()
        meas_freq_est = np.linspace(6.4, 6.6, self.num_qubits).tolist()
        pulse_library = [
            PulseLibraryItem(name="pulse_1", samples=[[0.0, 0.0], [0.0, 0.1]]),
            PulseLibraryItem(name="pulse_2", samples=[[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]),
            PulseLibraryItem(
                name="pulse_3", samples=[[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]]
            ),
        ]

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
    generate a fake backend from a combination of random defaults
    (with a fixable ``seed``) driven from a series of optional input arguments.
    """

    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str],
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_gates: list[str] | None = None,
        seed: int = 42,
    ):
        """
        Args:
            basis_gates (list[str]): List of basis gate names to be supported by
                the backend. The currently supported gate names are:
                ``"cx"``, ``"ecr"``, ``"id"``, ``"rz"``, ``"sx"``, and ``"x"``
                Common sets of basis gates are ``["cx", "id", "rz", "sx", "x"]``
                and ``["ecr", "id", "rz", "sx", "x"]``.

            num_qubits (int): Number of qubits for the fake backend.

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

            control_flow (bool): Flag to enable control flow directives on the backend
                (defaults to False).

            calibrate_gates (list[str] | None): List of gate names which should contain
                default calibration entries (overriden if an ``instruction_schedule_map`` is
                provided). These must be a subset of ``basis_gates``.

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
        # (duplicating a bit of logic)
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
            calibrate_gates,
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
