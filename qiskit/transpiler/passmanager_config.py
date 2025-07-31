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


class PassManagerConfig:
    """Pass Manager Configuration."""

    def __init__(
        self,
        initial_layout=None,
        basis_gates=None,
        coupling_map=None,
        layout_method=None,
        routing_method=None,
        translation_method=None,
        scheduling_method=None,
        instruction_durations=None,
        approximation_degree=None,
        seed_transpiler=None,
        timing_constraints=None,
        unitary_synthesis_method="default",
        unitary_synthesis_plugin_config=None,
        target=None,
        hls_config=None,
        init_method=None,
        optimization_method=None,
        qubits_initially_zero=True,
    ):
        """Initialize a PassManagerConfig object

        Args:
            initial_layout (Layout): Initial position of virtual qubits on
                physical qubits.
            basis_gates (list): List of basis gate names to unroll to.
            coupling_map (CouplingMap): Directed graph represented a coupling
                map.
            layout_method (str): the pass to use for choosing initial qubit
                placement. This will be the plugin name if an external layout stage
                plugin is being used.
            routing_method (str): the pass to use for routing qubits on the
                architecture. This will be a plugin name if an external routing stage
                plugin is being used.
            translation_method (str): the pass to use for translating gates to
                basis_gates. This will be a plugin name if an external translation stage
                plugin is being used.
            scheduling_method (str): the pass to use for scheduling instructions. This will
                be a plugin name if an external scheduling stage plugin is being used.
            instruction_durations (InstructionDurations): Dictionary of duration
                (in dt) for each instruction.
            approximation_degree (float): heuristic dial used for circuit approximation
                (1.0=no approximation, 0.0=maximal approximation)
            seed_transpiler (int): Sets random seed for the stochastic parts of
                the transpiler.
            timing_constraints (TimingConstraints): Hardware time alignment restrictions.
            unitary_synthesis_method (str): The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method. You can see a list of
                installed plugins with :func:`.unitary_synthesis_plugin_names`.
            target (Target): The backend target
            hls_config (HLSConfig): An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.
            init_method (str): The plugin name for the init stage plugin to use
            optimization_method (str): The plugin name for the optimization stage plugin
                to use.
            qubits_initially_zero (bool): Indicates whether the input circuit is
                zero-initialized.
        """
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.init_method = init_method
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.optimization_method = optimization_method
        self.scheduling_method = scheduling_method
        self.instruction_durations = instruction_durations
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target
        self.hls_config = hls_config
        self.qubits_initially_zero = qubits_initially_zero
        # Stores whether the basis gates are Clifford+T,
        # in which case we use stage manager plugins adapted to Clifford+T.
        self._is_clifford_t = False

    @classmethod
    def from_backend(cls, backend, _skip_target=False, **pass_manager_options):
        """Construct a configuration based on a backend and user input.

        This method automatically generates a PassManagerConfig object based on the backend's
        features. User options can be used to overwrite the configuration.

        Args:
            backend (BackendV2): The backend that provides the configuration.
            pass_manager_options: User-defined option-value pairs.

        Returns:
            PassManagerConfig: The configuration generated based on the arguments.

        Raises:
            AttributeError: If the backend does not support a `configuration()` method.
        """
        res = cls(**pass_manager_options)
        if res.basis_gates is None:
            res.basis_gates = backend.operation_names
        if res.coupling_map is None:
            res.coupling_map = backend.coupling_map
        if res.instruction_durations is None:
            res.instruction_durations = backend.instruction_durations
        if res.target is None and not _skip_target:
            res.target = backend.target
        if res.scheduling_method is None and hasattr(backend, "get_scheduling_stage_plugin"):
            res.scheduling_method = backend.get_scheduling_stage_plugin()
        if res.translation_method is None and hasattr(backend, "get_translation_stage_plugin"):
            res.translation_method = backend.get_translation_stage_plugin()
        return res

    def __str__(self):
        newline = "\n"
        newline_tab = "\n\t"
        return (
            "Pass Manager Config:\n"
            f"\tinitial_layout: {self.initial_layout}\n"
            f"\tbasis_gates: {self.basis_gates}\n"
            f"\tcoupling_map: {self.coupling_map}\n"
            f"\tlayout_method: {self.layout_method}\n"
            f"\trouting_method: {self.routing_method}\n"
            f"\ttranslation_method: {self.translation_method}\n"
            f"\tscheduling_method: {self.scheduling_method}\n"
            f"\tinstruction_durations: {str(self.instruction_durations).replace(newline, newline_tab)}\n"
            f"\tapproximation_degree: {self.approximation_degree}\n"
            f"\tseed_transpiler: {self.seed_transpiler}\n"
            f"\ttiming_constraints: {self.timing_constraints}\n"
            f"\tunitary_synthesis_method: {self.unitary_synthesis_method}\n"
            f"\tunitary_synthesis_plugin_config: {self.unitary_synthesis_plugin_config}\n"
            f"\tqubits_initially_zero: {self.qubits_initially_zero}\n"
            f"\ttarget: {str(self.target).replace(newline, newline_tab)}\n"
        )
