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

"""
====================================================================
Synthesis Plugins (:mod:`qiskit.transpiler.passes.synthesis.plugin`)
====================================================================

.. currentmodule:: qiskit.transpiler.passes.synthesis.plugin

This module defines the plugin interfaces for the synthesis transpiler passes
in Qiskit. These provide a hook point for external python packages to implement
their own synthesis techniques and have them seamlessly exposed as opt-in
options to users when they run :func:`~qiskit.compiler.transpile`.

The plugin interfaces are built using setuptools
`entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`__
which enable packages external to qiskit to advertise they include a synthesis
plugin.

Writing Plugins
===============

Unitary Synthesis Plugins
-------------------------

To write a unitary synthesis plugin there are 2 main steps. The first step is
to create a subclass of the abstract plugin class:
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`.
The plugin class defines the interface and contract for unitary synthesis
plugins. The primary method is
:meth:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin.run`
which takes in a single positional argument, a unitary matrix as a numpy array,
and is expected to return a :class:`~qiskit.dagcircuit.DAGCircuit` object
representing the synthesized circuit from that unitary matrix. Then to inform
the Qiskit transpiler about what information is necessary for the pass there
are several required property methods that need to be implemented,
``supports_basis_gates``, ``supports_coupling_map``, and
``supports_approximation_degree`` which return either ``True`` or ``False``
depending on whether the plugin supports and/or requires that input to perform
synthesis. An example plugin class would look something like::

    from qiskit.transpiler.passes.synthesis import plugin
    from qiskit_plugin_pkg.synthesis import generate_dag_circuit_from_matrix


    class SpecialUnitarySynthesis(plugin.UnitarySynthesisPlugin):
        @property
        def supports_basis_gates(self):
            return True

        @property
        def supports_coupling_map(self):
            return False

        @property
        def supports_approximation_degree(self):
            return False

        @property
        def supports_natural_direction(self):
            return False

        @property
        def supports_pulse_optimize(self):
            return False

        @property
        def supports_gate_lengths(self):
            return False

        @property
        def supports_gate_errors(self):
            return False

        def run(self, unitary, **options):
            basis_gates = options['basis_gates']
            dag_circuit = generate_dag_circuit_from_matrix(unitary, basis_gates)
            return dag_circuit

If for some reason the available inputs to the
:meth:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin.run`
method are insufficient please open an issue and we can discuss expanding the
plugin interface with new opt-in inputs that can be added in a backwards
compatible manner for future releases. Do note though that this plugin interface
is considered stable and guaranteed to not change in a breaking manner. If
changes are needed (for example to expand the available optional input options)
it will be done in a way that will **not** require changes from existing
plugins.

.. note::

    All methods prefixed with ``supports_`` are reserved on a
    ``UnitarySynthesisPlugin`` derived class for part of the interface. You
    should not define any custom ``supports_*`` methods on a subclass that
    are not defined in the abstract class.


The second step is to expose the
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin` as
a setuptools entry point in the package metadata. This is done by simply adding
an ``entry_points`` entry to the ``setuptools.setup`` call in the ``setup.py``
for the plugin package with the necessary entry points under the
``qiskit.unitary_synthesis`` namespace. For example::

    entry_points = {
        'qiskit.unitary_synthesis': [
            'special = qiskit_plugin_pkg.module.plugin:SpecialUnitarySynthesis',
        ]
    },

(note that the entry point ``name = path`` is a single string not a Python
expression). There isn't a limit to the number of plugins a single package can
include as long as each plugin has a unique name. So a single package can
expose multiple plugins if necessary. The name ``default`` is used by Qiskit
itself and can't be used in a plugin.

Using Plugins
=============

To use a plugin all you need to do is install the package that includes a
synthesis plugin. Then Qiskit will automatically discover the installed
plugins and expose them as valid options for the appropriate
:func:`~qiskit.compiler.transpiler` kwargs and pass constructors. If there are
any installed plugins which can't be loaded/imported this will be logged to
Python logging.

To get the installed list of installed unitary synthesis plugins you can use the
:func:`qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
function.

Plugin API
==========

Unitary Synthesis Plugins
-------------------------

.. autosummary::
   :toctree: ../stubs/

   UnitarySynthesisPlugin
   UnitarySynthesisPluginManager
   unitary_synthesis_plugin_names

"""

import abc

import stevedore


class UnitarySynthesisPlugin(abc.ABC):
    """Abstract plugin Synthesis plugin class

    This abstract class defines the interface for unitary synthesis plugins.
    """

    @property
    @abc.abstractmethod
    def supports_basis_gates(self):
        """Return whether the plugin supports taking ``basis_gates``

        If this returns ``True`` the plugin's ``run()`` method will be
        passed a ``basis_gates`` kwarg with a list of gate names the target
        backend supports. For example, ``['sx', 'x', 'cx', 'id', 'rz']``."""
        pass

    @property
    @abc.abstractmethod
    def supports_coupling_map(self):
        """Return whether the plugin supports taking ``coupling_map``

        If this returns ``True`` the plugin's ``run()`` method will receive
        two kwargs ``coupling_map`` and ``qubits``. The ``coupling_map`` kwarg
        will be a :class:`~qiskit.transpiler.CouplingMap` object representing
        the qubit connectivity of the target backend. The ``qubits`` kwarg will
        receive a list of integers that represent the qubit indices in the
        coupling map that unitary is on.
        """
        pass

    @property
    @abc.abstractmethod
    def supports_approximation_degree(self):
        """Return whether the plugin supports taking ``approximation_degree``

        If this returns ``True`` the plugin's ``run()`` method will receive
        a ``approximation_degree`` kwarg with a float value between 0 and 1
        representing the closeness of the approximation to use (0: lowest,
        1: highest).
        """
        pass

    @property
    @abc.abstractmethod
    def supports_natural_direction(self):
        """Return whether the plugin supports a toggle for considering
        directionality of 2-qubit gates as ``natural_direction``.

        Refer to the documentation for :class:`~qiskit.transpiler.passes.UnitarySynthesis`
        for the possible values and meaning of these values.
        """
        pass

    @property
    @abc.abstractmethod
    def supports_pulse_optimize(self):
        """Return whether the plugin supports a toggle to optimize pulses
        during synthesis as ``pulse_optimize``.

        Refer to the documentation for :class:`~qiskit.transpiler.passes.UnitarySynthesis`
        for the possible values and meaning of these values.
        """
        pass

    @property
    @abc.abstractmethod
    def supports_gate_lengths(self):
        """Return whether the plugin supports taking ``gate_lengths``

        ``gate_lengths`` will be a dictionary in the form of
        ``{gate_name: {(qubit_1, qubit_2): length}}``. For example::

            {
            'sx': {(0,): 0.0006149355812506126, (1,): 0.0006149355812506126},
            'cx': {(0, 1): 0.012012477900732316, (1, 0): 5.191111111111111e-07}
            }

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate lengths on every
        gate for each qubit.
        """
        pass

    @property
    @abc.abstractmethod
    def supports_gate_errors(self):
        """Return whether the plugin supports taking ``gate_errors``

        ``gate_errors`` will be a dictionary in the form of
        ``{gate_name: {(qubit_1, qubit_2): error}}``. For example::

            {
            'sx': {(0,): 0.0006149355812506126, (1,): 0.0006149355812506126},
            'cx': {(0, 1): 0.012012477900732316, (1, 0): 5.191111111111111e-07}
            }

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate errors on every
        gate for each qubit.
        """
        pass

    @abc.abstractmethod
    def run(self, unitary, **options):
        """Run synthesis for the given unitary matrix

        Args:
            unitary (numpy.ndarray): The unitary matrix to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object
            options: The optional kwargs that are passed based on the output
                of :meth:`supports_basis_gates`, :meth:`supports_coupling_map`,
                and :meth:`supports_approximation_degree`. If
                :meth:`supports_coupling_map` returns ``True`` a kwarg
                ``coupling_map`` will be passed either containing ``None`` (if
                there is no coupling map) or a
                :class:`~qiskit.transpiler.CouplingMap` object. If
                :meth:`supports_basis_gates` returns ``True`` then a kwarg
                ``basis_gates`` will the list of basis gate names will be
                passed. Finally if :meth:`supports_approximation_degree`
                returns ``True`` a kwarg ``approximation_degree`` containing
                a float for the approximation value will be passed.

        Returns:
            DAGCircuit: The dag circuit representation of the unitary. Alternatively,
                you can return a tuple of the form ``(dag, wires)`` where ``dag`` is the dag
                circuit representation of the circuit representation of the unitary
                and ``wires`` is the mapping wires to use for
                :meth:`qiskit.dagcircuit.DAGCircuit.substitute_node_with_dag`.
        """
        pass


class UnitarySynthesisPluginManager:
    """Unitary Synthesis plugin manager class

    This class tracks the installed plugins, it has a single property,
    ``ext_plugins`` which contains a list of stevedore plugin objects.
    """

    def __init__(self):
        self.ext_plugins = stevedore.ExtensionManager(
            "qiskit.unitary_synthesis", invoke_on_load=True, propagate_map_exceptions=True
        )


def unitary_synthesis_plugin_names():
    """Return a list of installed unitary synthesis plugin names

    Returns:
        list: A list of the installed unitary synthesis plugin names. The
            plugin names are valid values for the
            :func:`~qiskit.compiler.transpile` kwarg
            ``unitary_synthesis_method``.
    """
    plugins = UnitarySynthesisPluginManager()
    return plugins.ext_plugins.names()
