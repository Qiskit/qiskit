# -*- coding: utf-8 -*-
# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""PassManager class for the transpiler."""

from functools import partial
from ._propertyset import PropertySet
from ._basepasses import BasePass
from ._fencedobjs import FencedPropertySet, FencedDAGCircuit
from ._transpilererror import TranspilerError
from ._propertysetutilities import fixed_point


class PassManager():
    """ A PassManager schedules the passes """

    def __init__(self, ignore_requires=None, ignore_preserves=None, idempotence=None,
                 max_iteration=None):
        """
        Initialize an empty PassManager object (with no passes scheduled).

        Args:
            ignore_requires (bool): The schedule ignores the request field in the passes. The
                default setting in the pass is False.
            ignore_preserves (bool): The schedule ignores the preserves field in the passes.  The
                default setting in the pass is False.
            idempotence (bool): The schedule considers every pass idempotent.
                The default setting in the pass is True.
            max_iteration (int): The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """

        self.working_list = WorkingList()
        self.property_set = PropertySet()
        self.fenced_property_set = FencedPropertySet(self.property_set)
        self.valid_passes = set()
        self.pass_options = {'ignore_requires': ignore_requires,
                             'ignore_preserves': ignore_preserves,
                             'idempotence': idempotence,
                             'max_iteration': max_iteration}
        # Add default utilities
        self.property_set.add_utility(fixed_point)

    def __getitem__(self, key):
        return self.property_set[key]

    def __setitem__(self, key, value):
        self.property_set[key] = value

    def _join_options(self, passset_options, pass_options):
        # Remove Nones
        passmanager_level = {k: v for k, v in self.pass_options.items() if v is not None}
        passset_level = {k: v for k, v in passset_options.items() if v is not None}
        pass_level = {k: v for k, v in pass_options.items() if v is not None}
        return {**passmanager_level, **passset_level, **pass_level}

    def add_pass(self, pass_or_list_of_passes, idempotence=None, ignore_requires=None,
                 ignore_preserves=None, max_iteration=None, **control_flow_plugins):
        """
        Args:
            pass_or_list_of_passes (TransformationPass or AnalysisPass or list): algo
            idempotence (bool): algo Default: True
            ignore_preserves (bool): algo Default: False
            ignore_requires (bool): algo Default: False
            max_iteration (int): Limit in the amount of iterations  Default: 1000
            control_flow_plugins (kwargs): See add_control_flow_plugin(): Dictionary of control flow
                plugins. Default:
                do_while (callable property_set -> boolean): The pass (or passes) repeats until the
                   callable returns False.
                   Default: lambda x: False # i.e. pass_or_list_of_passes runs once
                condition (callable property_set -> boolean): The pass (or passes) runs only if the
                   callable returns True.
                   Default: lambda x: True # i.e. pass_or_list_of_passes runs
        Raises:
            TranspilerError: if pass_or_list_of_passes is not a proper pass.
        """

        passset_options = {'idempotence': idempotence,
                           'ignore_requires': ignore_requires,
                           'ignore_preserves': ignore_preserves,
                           'max_iteration': max_iteration}

        if isinstance(pass_or_list_of_passes, BasePass):
            pass_or_list_of_passes = [pass_or_list_of_passes]

        for pass_ in pass_or_list_of_passes:
            if isinstance(pass_, BasePass):
                pass_.set(**self._join_options(passset_options, pass_._settings))
            else:
                raise TranspilerError('%s is not a pass instance' % pass_.__class__)

        for name, plugin in control_flow_plugins.items():
            if callable(plugin):
                control_flow_plugins[name] = partial(plugin, self.fenced_property_set)

        self.working_list.add(pass_or_list_of_passes, **control_flow_plugins)

    def run_passes(self, dag):
        """Run all the passes on the dag.

        Args:
            dag (DAGCircuit): dag circuit to transform via all the registered passes
        """
        for pass_ in self.working_list:
            self._do_pass(pass_, dag)

    def _do_pass(self, pass_, dag):
        """
        Do a pass and its "requires".
        Args:
            pass_ (BasePass): Pass to do.
            dag (DAGCircuit): The dag in which the pass is ran.
        Raises:
            TranspilerError: If the pass is not a proper pass instance.
        """

        # First, do the requires of pass_
        if not pass_.ignore_requires:
            for required_pass in pass_.requires:
                self._do_pass(required_pass, dag)

        # Run the pass itself, if not already ran (exists in valid_passes)
        if pass_ not in self.valid_passes:
            if pass_.is_TransformationPass:
                pass_.property_set = self.fenced_property_set
                pass_.run(dag)
            elif pass_.is_AnalysisPass:
                pass_.property_set = self.property_set
                pass_.run(FencedDAGCircuit(dag))
            else:
                raise TranspilerError("I dont know how to handle this type of pass")

            # update the valid_passes property
            self._update_valid_passes(pass_)

    def _update_valid_passes(self, pass_):
        if not pass_.is_AnalysisPass:  # Analysis passes preserve all
            if pass_.ignore_preserves:
                self.valid_passes.clear()
            else:
                self.valid_passes.intersection_update(set(pass_.preserves))

        if pass_.idempotence:
            self.valid_passes.add(pass_)

    def add_control_flow_plugin(self, name, control_flow_plugin):
        """
        Adds a control flow plugin.
        Args:
            name (string): Name of the plugin.
            control_flow_plugin (ControlFlowPlugin): The class implementing a control flow plugin.
        """
        self.working_list.add_control_flow_plugin(name, control_flow_plugin)

    def remove_control_flow_plugin(self, name):
        """
        Removes a control flow plugin.
        Args:
            name:
        """
        self.working_list.remove_control_flow_plugin(name)


class WorkingList():
    """
    A working list is the way that a pass manager organize the things to do.
    """

    def __init__(self):
        self.list_of_items = []
        self.control_flow_plugins = {'condition': PluginConditional,
                                     'do_while': PluginDoWhile, }

    def add_control_flow_plugin(self, name, control_flow_plugin):
        """
        Adds a control flow plugin.
        Args:
            name (string): Name of the plugin.
            control_flow_plugin (ControlFlowPlugin): The class implementing a control flow plugin.
        """
        self.control_flow_plugins[name] = control_flow_plugin

    def remove_control_flow_plugin(self, name):
        """
        Removes the plugin called name.

        Args:
            name (string): The control flow plugin to remove.
        Raises:
            KeyError: If the name is not found.
        """
        if name not in self.control_flow_plugins:
            raise KeyError("Control flow plugin not found: %s" % name)
        del self.control_flow_plugins[name]

    def add(self, passes, **control_flow_plugins):
        """
        Populates the working list with passes.

        Args:
            passes (list): a list of passes to add to the working list.
            control_flow_plugins (kwargs): See add_control_flow_plugin(). Dictionary of control flow
                plugins. Defaults:
                do_while (callable property_set -> boolean): The pass (or passes) repeats until the
                   callable returns False.
                   Default: lambda x: False # i.e. pass_or_list_of_passes runs once
                condition (callable property_set -> boolean): The pass (or passes) runs only if the
                   callable returns True.
                   Default: lambda x: True # i.e. pass_or_list_of_passes runs
        """
        for control_flow, condition in control_flow_plugins.items():
            if condition and control_flow in self.control_flow_plugins:
                self.list_of_items.append(
                    self.control_flow_plugins[control_flow](passes, **control_flow_plugins))
                break
        else:
            self.list_of_items.append(passes)

    def __iter__(self):
        for item in self.list_of_items:
            for pass_ in item:
                yield pass_


class ControlFlowPlugin():
    """This class is a base class for multiple types of working list. When you iterate on it, it
    returns the next pass to run. """

    def __init__(self, passes, **control_flow_plugins):
        self.working_list = WorkingList()
        self.working_list.add(passes, **control_flow_plugins)

    def __iter__(self):
        raise NotImplementedError


class PluginDoWhile(ControlFlowPlugin):
    """This type of working list item implements a set of passes in a do while loop. """

    def __init__(self, passes, do_while=None, **_):  # pylint: disable=super-init-not-called
        self.do_while = do_while
        self.max_iteration = min([pass_.max_iteration for pass_ in passes])
        super().__init__(passes)

    def __iter__(self):
        iteration = 0
        while True:
            for pass_ in self.working_list:
                yield pass_
            iteration += 1
            if iteration >= self.max_iteration:
                raise TranspilerError("Maximum iteration reached. max_iteration=%i" %
                                      self.max_iteration)
            if not self.do_while():
                break


class PluginConditional(ControlFlowPlugin):
    """This type of working list item implements a set of passes under certain condition. """

    def __init__(self, passes, do_while=None, condition=None,
                 **control_flow_plugins):  # pylint: disable=super-init-not-called
        self.condition = condition
        super().__init__(passes, do_while=do_while, **control_flow_plugins)

    def __iter__(self):
        if self.condition():
            for pass_ in self.working_list:
                yield pass_
