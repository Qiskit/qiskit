# -*- coding: utf-8 -*-

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

"""Pass Manager Configuraiton class."""


class PassManagerConfig:
    """Pass Manager Configuration.
    """

    def __init__(self,
                 initial_layout=None,
                 basis_gates=None,
                 coupling_map=None,
                 layout_method=None,
                 routing_method=None,
                 translation_method=None,
                 backend_properties=None,
                 seed_transpiler=None):
        """Initialize a PassManagerConfig object

        Args:
            initial_layout (Layout): Initial position of virtual qubits on
                physical qubits.
            basis_gates (list): List of basis gate names to unroll to.
            coupling_map (CouplingMap): Directed graph represented a coupling
                map.
            layout_method (str): the pass to use for choosing initial qubit
                placement.
            routing_method (str): the pass to use for routing qubits on the
                architecture.
            translation_method (str): the pass to use for translating gates to
                basis_gates.
            backend_properties (BackendProperties): Properties returned by a
                backend, including information on gate errors, readout errors,
                qubit coherence times, etc.
            seed_transpiler (int): Sets random seed for the stochastic parts of
                the transpiler.
        """
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.backend_properties = backend_properties
        self.seed_transpiler = seed_transpiler
