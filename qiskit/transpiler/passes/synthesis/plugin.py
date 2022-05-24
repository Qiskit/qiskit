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
are several required property methods that need to be implemented such as
``supports_basis_gates`` and ``supports_coupling_map`` depending on whether the
plugin supports and/or requires that input to perform synthesis. For the full
details refer to the
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`
documentation for all the required fields. An example plugin class would look
something like::

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

        @property
        def min_qubits(self):
            return None

        @property
        def max_qubits(self):
            return None

        @property
        def supported_bases(self):
            return None

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

Unitary Synthesis Plugin Configuration
''''''''''''''''''''''''''''''''''''''

For some unitary synthesis plugins that expose multiple options and tunables
the plugin interface has an option for users to provide a free form
configuration dictionary. This will be passed through to the ``run()`` method
as the ``config`` kwarg. If your plugin has these configuration options you
should clearly document how a user should specify these configuration options
and how they're used as it's a free form field.

Using Plugins
=============

To use a plugin all you need to do is install the package that includes a
synthesis plugin. Then Qiskit will automatically discover the installed
plugins and expose them as valid options for the appropriate
:func:`~qiskit.compiler.transpile` kwargs and pass constructors. If there are
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
    """Abstract unitary synthesis plugin class

    This abstract class defines the interface for unitary synthesis plugins.
    """

    @property
    @abc.abstractmethod
    def max_qubits(self):
        """Return the maximum number of qubits the unitary synthesis plugin supports.

        If the size of the unitary to be synthesized exceeds this value the
        ``default`` plugin will be used. If there is no upper bound return
        ``None`` and all unitaries (``>= min_qubits`` if it's defined) will be
        passed to this plugin when it's enabled.
        """
        pass

    @property
    @abc.abstractmethod
    def min_qubits(self):
        """Return the minimum number of qubits the unitary synthesis plugin supports.

        If the size of the unitary to be synthesized is below this value the
        ``default`` plugin will be used. If there is no lower bound return
        ``None`` and all unitaries (``<= max_qubits`` if it's defined) will be
        passed to this plugin when it's enabled.
        """
        pass

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
        one kwarg ``coupling_map``. The ``coupling_map`` kwarg will be set to a
        tuple with the first element being a
        :class:`~qiskit.transpiler.CouplingMap` object representing the qubit
        connectivity of the target backend, the second element will be a list
        of integers that represent the qubit indices in the coupling map that
        unitary is on. Note that if the target backend doesn't have a coupling
        map set, the ``coupling_map`` kwarg's value will be ``(None, qubit_indices)``.
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

        where the ``length`` value is in units of seconds.

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
        gate for each qubit. The gate error rates reported in ``gate_errors``
        are provided by the target device ``Backend`` object and the exact
        meaning might be different depending on the backend.
        """
        pass

    @property
    @abc.abstractmethod
    def supported_bases(self):
        """Returns a dictionary of supported bases for synthesis

        This is expected to return a dictionary where the key is a string
        basis and the value is a list of gate names that the basis works in.
        If the synthesis method doesn't support multiple bases this should
        return ``None``. For example::

            {
                "XZX": ["rz", "rx"],
                "XYX": ["rx", "ry"],
            }

        If a dictionary is returned by this method the run kwargs will be
        passed a parameter ``matched_basis`` which contains a list of the
        basis strings (i.e. keys in the dictionary) which match the target basis
        gate set for the transpilation. If no entry in the dictionary matches
        the target basis gate set then the ``matched_basis`` kwarg will be set
        to an empty list, and a plugin can choose how to deal with the target
        basis gate set not matching the plugin's capabilities.
        """
        pass

    @property
    def supports_target(self):
        """Whether the plugin supports taking ``target`` as an option

        ``target`` will be a :class:`~.Target` object representing the target
        device for the output of the synthesis pass.

        By default this will be ``False`` since the plugin interface predates
        the :class:`~.Target` class. If a plugin returns ``True`` for this
        attribute, it is expected that the plugin will use the
        :class:`~.Target` instead of the values passed if any of
        ``supports_gate_lengths``, ``supports_gate_errors``,
        ``supports_coupling_map``, and ``supports_basis_gates`` are set
        (although ideally all those parameters should contain duplicate
        information).
        """
        return False

    @abc.abstractmethod
    def run(self, unitary, **options):
        """Run synthesis for the given unitary matrix

        Args:
            unitary (numpy.ndarray): The unitary matrix to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object
            options: The optional kwargs that are passed based on the output
                the ``support_*`` methods on the class. Refer to the
                documentation for these methods on
                :class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`
                to see what the keys and values are.

        Returns:
            DAGCircuit: The dag circuit representation of the unitary. Alternatively, you can return
            a tuple of the form ``(dag, wires)`` where ``dag`` is the dag circuit representation of
            the circuit representation of the unitary and ``wires`` is the mapping wires to use for
            :meth:`qiskit.dagcircuit.DAGCircuit.substitute_node_with_dag`. If you return a tuple
            and ``wires`` is ``None`` this will behave just as if only a
            :class:`~qiskit.dagcircuit.DAGCircuit` was returned. Additionally if this returns
            ``None`` no substitution will be made.

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
        list: A list of the installed unitary synthesis plugin names. The plugin names are valid
        values for the :func:`~qiskit.compiler.transpile` kwarg ``unitary_synthesis_method``.
    """
    # NOTE: This is not a shared global instance to avoid an import cycle
    # at load time for the default plugin.
    plugins = UnitarySynthesisPluginManager()
    return plugins.ext_plugins.names()
