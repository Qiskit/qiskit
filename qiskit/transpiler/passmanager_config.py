# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass Manager Configuration class."""

from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.instruction_durations import InstructionDurations


class PassManagerConfig:
    """Pass Manager Configuration."""

    def __init__(
        self,
        initial_layout=None,
        basis_gates=None,
        inst_map=None,
        coupling_map=None,
        layout_method=None,
        routing_method=None,
        translation_method=None,
        scheduling_method=None,
        instruction_durations=None,
        backend_properties=None,
        approximation_degree=None,
        seed_transpiler=None,
        timing_constraints=None,
        unitary_synthesis_method="default",
        unitary_synthesis_plugin_config=None,
        target=None,
    ):
        """Initialize a PassManagerConfig object

        Args:
            initial_layout (Layout): Initial position of virtual qubits on
                physical qubits.
            basis_gates (list): List of basis gate names to unroll to.
            inst_map (InstructionScheduleMap): Mapping object that maps gate to schedule.
            coupling_map (CouplingMap): Directed graph represented a coupling
                map.
            layout_method (str): the pass to use for choosing initial qubit
                placement.
            routing_method (str): the pass to use for routing qubits on the
                architecture.
            translation_method (str): the pass to use for translating gates to
                basis_gates.
            scheduling_method (str): the pass to use for scheduling instructions.
            instruction_durations (InstructionDurations): Dictionary of duration
                (in dt) for each instruction.
            backend_properties (BackendProperties): Properties returned by a
                backend, including information on gate errors, readout errors,
                qubit coherence times, etc.
            approximation_degree (float): heuristic dial used for circuit approximation
                (1.0=no approximation, 0.0=maximal approximation)
            seed_transpiler (int): Sets random seed for the stochastic parts of
                the transpiler.
            timing_constraints (TimingConstraints): Hardware time alignment restrictions.
            unitary_synthesis_method (str): The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method.
            target (Target): The backend target
        """
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.inst_map = inst_map
        self.coupling_map = coupling_map
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.scheduling_method = scheduling_method
        self.instruction_durations = instruction_durations
        self.backend_properties = backend_properties
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target

    @classmethod
    def from_backend(cls, backend, **pass_manager_options):
        """Construct a configuration based on a backend and user input.

        This method automatically gererates a PassManagerConfig object based on the backend's
        features. User options can be used to overwrite the configuration.

        Args:
            backend (BackendV1): The backend that provides the configuration.
            pass_manager_options: User-defined option-value pairs.

        Returns:
            PassManagerConfig: The configuration generated based on the arguments.

        Raises:
            AttributeError: If the backend does not support a `configuration()` method.
        """
        res = cls(**pass_manager_options)
        config = backend.configuration()

        if res.basis_gates is None:
            res.basis_gates = getattr(config, "basis_gates", None)
        if res.inst_map is None and hasattr(backend, "defaults"):
            res.inst_map = backend.defaults().instruction_schedule_map
        if res.coupling_map is None:
            res.coupling_map = CouplingMap(getattr(config, "coupling_map", None))
        if res.instruction_durations is None:
            res.instruction_durations = InstructionDurations.from_backend(backend)
        if res.backend_properties is None:
            res.backend_properties = backend.properties()
        if res.target is None:
            backend_version = getattr(backend, "version", 0)
            if not isinstance(backend_version, int):
                backend_version = 0
            if backend_version >= 2:
                res.target = backend.target

        return res
