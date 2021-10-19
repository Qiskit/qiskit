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

from qiskit.exceptions import QiskitError
from qiskit.transpiler.coupling import CouplingMap


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

    @staticmethod
    def from_backend(backend, **pass_manager_options):
        """Construct a configuration based on a backend and user input.

        This method returns a PassManagerConfig object with the option values set by
        the user. For those options not specified in the arguments, the method looks
        for them in the backend's configuration. If still not found, then a default
        value (usually `None`) is used.

        Args:
            backend: The backend that provides the configuration.
            pass_manager_options: User-defined option-value pairs.

        Returns:
            PassManagerConfig: The configuration generated based on the arguments.

        Raises:
            QiskitError: If the backend does not support a `configuration()` method
                or the configuration does not support a `to_dict()` method.
            AttributeError: If the field passed in is not part of the options.
        """
        res = PassManagerConfig()
        try:
            backend_dict = backend.configuration().to_dict()
        except:
            raise QiskitError("Invalid backend type %s" % type(backend))

        def get_config(option, default=None):
            """Helper function that returns the value of a specified option."""
            if option in pass_manager_options:
                return pass_manager_options.pop(option)
            return backend_dict.get(option, default)

        res.initial_layout = get_config("initial_layout")
        res.basis_gates = get_config("basis_gates")
        res.inst_map = get_config("inst_map")
        res.coupling_map = CouplingMap(get_config("coupling_map"))
        res.layout_method = get_config("layout_method")
        res.routing_method = get_config("routing_method")
        res.translation_method = get_config("translation_method")
        res.scheduling_method = get_config("scheduling_method")
        res.instruction_durations = get_config("instruction_durations")
        res.backend_properties = get_config("backend_properties")
        res.approximation_degree = get_config("approximation_degree")
        res.seed_transpiler = get_config("seed_transpiler")
        res.timing_constraints = get_config("timing_constraints")
        res.unitary_synthesis_method = get_config("unitary_synthesis_method", "default")

        # Raise an error if the user specifies an option that is not a member of PassManagerConfig
        if pass_manager_options:
            raise AttributeError(f"Option {pass_manager_options.popitem()[0]} is not defined")
        return res
