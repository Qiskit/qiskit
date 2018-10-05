# -*- coding: utf-8 -*-
# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""PassManager class for the transpiler."""

from functools import partial
from qiskit.dagcircuit import DAGCircuit
from ._propertyset import PropertySet
from ._basepasses import BasePass
from ._fencedobjs import FencedPropertySet, FencedDAGCircuit
from ._transpilererror import TranspilerError


class PassManager():
    """ A PassManager schedules the passes """

    def __init__(self, ignore_requires=None, ignore_preserves=None, max_iteration=None):
        """
        Initialize an empty PassManager object (with no passes scheduled).

        Args:
            ignore_requires (bool): The schedule ignores the requires field in the passes. The
                default setting in the pass is False.
            ignore_preserves (bool): The schedule ignores the preserves field in the passes. The
                default setting in the pass is False.
            max_iteration (int): The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.add_passes().
        self.working_list = []

        # global property set is the context of the circuit held by the pass manager
        # as it runs through its scheduled passes. Analysis passes may update the property_set,
        # but transformation passes have read-only access (via the fenced_property_set).
        self.property_set = PropertySet()
        self.fenced_property_set = FencedPropertySet(self.property_set)

        # passes already run that have not been invalidated
        self.valid_passes = set()

        # pass manager's overriding options for the passes it runs (for debugging)
        self.passmanager_options = {'ignore_requires': ignore_requires,
                                    'ignore_preserves': ignore_preserves,
                                    'max_iteration': max_iteration}
        from collections import defaultdict
        self.options = defaultdict(lambda: self.passmanager_options)

        # TODO
        ControlFlowPlugin.add_flow_controller('condition', PluginConditional)
        ControlFlowPlugin.add_flow_controller('do_while', PluginDoWhile)

    def _join_options(self, passset_options):
        """ Set the options of each individual pass, based on precedence rules:
        passmanager options (set via ``PassManager.__init__()``) override
        passset options (set via ``PassManager.add_passes()``), and those override
        pass options (set via ``BasePass.arg = value``).

        """
        default = {'ignore_preserves': False,  # Ignore preserves for this pass
                   'ignore_requires': False,  # Ignore requires for this pass
                   'max_iteration': 1000}  # Maximum allowed iteration on this pass

        passmanager_level = {k: v for k, v in self.passmanager_options.items() if v is not None}
        passset_level = {k: v for k, v in passset_options.items() if v is not None}
        return {**default, **passmanager_level, **passset_level}

    def add_passes(self, passes, ignore_requires=None, ignore_preserves=None, max_iteration=None,
                   **control_flow_plugins):
        """
        Args:
            passes (list[BasePass] or BasePass): pass(es) to be added to schedule
            ignore_preserves (bool): ignore the preserves claim of passes. Default: False
            ignore_requires (bool): ignore the requires need of passes. Default: False
            max_iteration (int): max number of iterations of passes. Default: 1000
            control_flow_plugins (kwargs): See add_control_flow_plugin(): Dictionary of
                control flow plugins. Default:
                do_while (callable property_set -> boolean): The passes repeat until the
                   callable returns False.
                   Default: lambda x: False # i.e. passes run once
                condition (callable property_set -> boolean): The passes run only if the
                   callable returns True.
                   Default: lambda x: True # i.e. passes run
        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
        """

        passset_options = {'ignore_requires': ignore_requires,
                           'ignore_preserves': ignore_preserves,
                           'max_iteration': max_iteration}

        options = self._join_options(passset_options)

        if isinstance(passes, BasePass):
            passes = [passes]

        for pass_ in passes:
            if not isinstance(pass_, BasePass):
                raise TranspilerError('%s is not a pass instance' % pass_.__class__)

        for name, plugin in control_flow_plugins.items():
            if callable(plugin):
                control_flow_plugins[name] = partial(plugin, self.fenced_property_set)
            else:
                raise TranspilerError('%s control-flow plugin is not callable' % name)

        self.working_list.append(
            ControlFlowPlugin.controller_factory(passes, options, **control_flow_plugins))

    def run_passes(self, dag):
        """Run all the passes on the dag.

        Args:
            dag (DAGCircuit): dag circuit to transform via all the registered passes
        """
        for passset in self.working_list:
            for pass_ in passset:
                dag = self._do_pass(pass_, dag, passset.options)

    def _do_pass(self, pass_, dag, options):
        """Do a pass and its "requires".

        Args:
            pass_ (BasePass): Pass to do.
            dag (DAGCircuit): The dag on which the pass is ran.
        Returns:
            DAGCircuit: The transformed dag in case of a transformation pass.
            The same input dag in case of an analysis pass.
        Raises:
            TranspilerError: If the pass is not a proper pass instance.
        """

        # First, do the requires of pass_
        if not options["ignore_requires"]:
            for required_pass in pass_.requires:
                self._do_pass(required_pass, dag, options)

        # Run the pass itself, if not already run
        if pass_ not in self.valid_passes:
            if pass_.is_transformation_pass:
                pass_.property_set = self.fenced_property_set
                new_dag = pass_.run(dag)
                if not isinstance(new_dag, DAGCircuit):
                    raise TranspilerError("Transformation passes should return a transformed dag."
                                          "The pass %s is returning a %s" % (type(pass_).__name__,
                                                                             type(new_dag)))
                dag = new_dag
            elif pass_.is_analysis_pass:
                pass_.property_set = self.property_set
                pass_.run(FencedDAGCircuit(dag))
            else:
                raise TranspilerError("I dont know how to handle this type of pass")

            # update the valid_passes property
            self._update_valid_passes(pass_, options['ignore_preserves'])

        return dag

    def _update_valid_passes(self, pass_, ignore_preserves):
        self.valid_passes.add(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            if ignore_preserves:
                self.valid_passes.clear()
            else:
                self.valid_passes.intersection_update(set(pass_.preserves))

    def add_control_flow_plugin(self, name, control_flow_plugin):
        """
        Adds a control flow plugin.
        Args:
            name (string): Name of the plugin.
            control_flow_plugin (ControlFlowPlugin): The class implementing a control flow plugin.
        """
        ControlFlowPlugin.add_flow_controller(name, control_flow_plugin)

    def remove_control_flow_plugin(self, name):
        """
        Removes a control flow plugin.
        Args:
            name:
        """
        ControlFlowPlugin.remove_flow_controller(name)


class ControlFlowPlugin():
    """This class is a base class for multiple types of working list. When you iterate on it, it
    returns the next pass to run. """

    # registered_flow_controllers= {'condition': PluginConditional,
    #                               'do_while': PluginDoWhile}
    registered_controllers = {}

    def __init__(self, passes, options, **control_flow_plugins):
        self.passes = ControlFlowPlugin.controller_factory(passes, options, **control_flow_plugins)
        self.options = options

    def __iter__(self):
        for pass_ in self.passes:
            yield pass_

    @classmethod
    def add_flow_controller(cls, name, control_flow_plugin):
        """
        Adds a control flow plugin.
        Args:
            name (string): Name of the plugin.
            control_flow_plugin (ControlFlowPlugin): The class implementing a control flow plugin.
        """
        cls.registered_controllers[name] = control_flow_plugin

    @classmethod
    def remove_flow_controller(cls, name):
        """
        Removes the plugin called name.

        Args:
            name (string): The control flow plugin to remove.
        Raises:
            KeyError: If the name is not found.
        """
        if name not in cls.registered_controllers:
            raise KeyError("Flow controller not found: %s" % name)
        del cls.registered_controllers[name]

    @classmethod
    def controller_factory(cls, passes, options, **control_flow_plugins):
        for control_flow, condition in control_flow_plugins.items():
            if condition and control_flow in cls.registered_controllers:
                return cls.registered_controllers[control_flow](passes, options,
                                                                **control_flow_plugins)
        return FlowControllerLinear(passes, options)


class FlowControllerLinear(ControlFlowPlugin):
    def __init__(self, passes, options):
        self.passes = passes
        self.options = options


class PluginDoWhile(ControlFlowPlugin):
    """This type of working list item implements a set of passes in a do while loop. """

    def __init__(self, passes, options, do_while=None,
                 **_):  # pylint: disable=super-init-not-called
        self.do_while = do_while
        self.max_iteration = options['max_iteration']
        super().__init__(passes, options)

    def __iter__(self):
        iteration = 0
        while True:
            for pass_ in self.passes:
                yield pass_
            iteration += 1
            if iteration >= self.max_iteration:
                raise TranspilerError("Maximum iteration reached. max_iteration=%i" %
                                      self.max_iteration)
            if not self.do_while():
                break


class PluginConditional(ControlFlowPlugin):
    """This type of working list item implements a set of passes under certain condition. """

    def __init__(self, passes, options, do_while=None, condition=None,
                 **control_flow_plugins):  # pylint: disable=super-init-not-called
        self.condition = condition
        super().__init__(passes, options, do_while=do_while, **control_flow_plugins)

    def __iter__(self):
        if self.condition():
            for pass_ in self.passes:
                yield pass_
