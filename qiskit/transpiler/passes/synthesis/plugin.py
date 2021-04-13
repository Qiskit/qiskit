# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import abc

import stevedore


class UnitarySynthesisPlugin(abc.ABC):
    """Abstract plugin Synthesis plugin class

    This abstract class defines the interface for unitary synthesis plugins.
    """

    @property
    @abc.abstractmethod
    def supports_basis_gates(self):
        """Return whether the plugin supports taking basis_gates"""
        pass

    @property
    @abc.abstractmethod
    def supports_coupling_map(self):
        """Return whether the plugin supports taking coupling_map"""
        pass

    @property
    @abc.abstractmethod
    def supports_approximation_degree(self):
        """Return whether the plugin supports taking approximation_degree"""
        pass

    @abc.abstractmethod
    def run(self, unitary, **options):
        """Run synthesis for the given unitary

        Args:
            unitary (numpy.ndarray): The unitary

        Returns:
            DAGCircuit: The dag circuit representation of the unitary
        """
        pass


class UnitarySynthesisPluginManager:

    def __init__(self):
        self.ext_plugins = stevedore.ExtensionManager(
            'qiskit.unitary_synthesis', invoke_on_load=True,
            propagate_map_exceptions=True)
