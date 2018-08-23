# -*- coding: utf-8 -*-
# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""PassManager class for the transpiler."""

from ._propertyset import PropertySet
from ._basepasses import BasePass
from functools import partial
from ._fencedobjs import FencedPropertySet, FencedDAGCircuit
from qiskit import QISKitError


class PassManager():
    """ A PassManager schedules the passes """

    def __init__(self, ignore_requires=None, ignore_preserves=None, idempotence=None):
        """
        Initialize an empty PassManager object (with no passes scheduled).

        Args:
            ignore_requires (bool): The schedule ignores the request field in the passes. The
                default setting in the pass is False.
            ignore_preserves (bool): The schedule ignores the preserves field in the passes.  The
                default setting in the pass is False.
            idempotence (bool): The schedule considers every pass idempotent.
                 The default setting in the pass is True.
        """

        self.working_list = WorkingList()
        self.property_set = PropertySet()
        self.ro_property_set = FencedPropertySet(self.property_set)
        self.valid_passes = set()
        self.pass_options = {'ignore_requires': ignore_requires,
                             'ignore_preserves': ignore_preserves,
                             'idempotence': idempotence}

    def __getitem__(self, key):
        return self.property_set[key]

    def __setitem__(self, key, value):
        self.property_set[key] = value

    def _join_options(self, passset_options, pass_options):
        # Remove Nones
        passmanager_level = {k: v for k, v in self.pass_options.items() if v is not None}
        passset_level = {k: v for k, v in passset_options.items() if v is not None}
        pass_level = {k: v for k, v in pass_options.items() if v is not None}
        return {**passmanager_level, **passset_level, **pass_level }

    def add_pass(self, pass_or_list_of_passes, do_while=None, condition=None,
                 idempotence=None, ignore_requires=None, ignore_preserves=None):
        """
        Args:
            pass_or_list_of_passes (pass instance or list):
            do_while (callable property_set -> boolean): The pass (or passes) repeats until the
               callable returns False.
               Default: lambda x: False # i.e. pass_or_list_of_passes runs once
            condition (callable property_set -> boolean): The pass (or passes) runs only if the
               callable returns True.
               Default: lambda x: True # i.e. pass_or_list_of_passes runs
        Raises:
            QISKitError: if pass_or_list_of_passes is not a proper pass.
        """

        passset_options = {'idempotence': idempotence,
                           'ignore_requires': ignore_requires,
                           'ignore_preserves': ignore_preserves}

        if isinstance(pass_or_list_of_passes, BasePass):
            pass_or_list_of_passes = [pass_or_list_of_passes]

        for pass_ in pass_or_list_of_passes:
            if isinstance(pass_, BasePass):
                pass_.set(**self._join_options(passset_options, pass_._settings))
            else:
                raise QISKitError('%s is not a pass instance' % pass_.__class__)

        if do_while:
            do_while = partial(do_while, self.property_set)

        if condition:
            condition = partial(condition, self.property_set)

        self.working_list.add(pass_or_list_of_passes, do_while, condition)

    def run_passes(self, dag):
        for pass_ in self.working_list:
            self._do_pass(pass_, dag)

    def _do_pass(self, pass_, dag):
        """
        Do a pass and its "requires".
        Args:
            pass_ (BasePass): Pass to do.
            dag (DAGCircuit): The dag in which the pass is ran.
        """

        # First, do the requires of pass_
        if not pass_.ignore_requires:
            for required_pass in pass_.requires:
                self._do_pass(required_pass, dag)

        # Run the pass itself, if not already ran (exists in valid_passes)
        if not pass_ in self.valid_passes:
            if pass_.isTransformationPass:
                pass_.run(dag, self.ro_property_set)
            elif pass_.isAnalysisPass:
                pass_.run(FencedDAGCircuit(dag), self.property_set)
            else:
                raise Exception("I dont know how to handle this type of pass")

            # update the valid_passes property
            self._update_valid_passes(pass_)

    def _update_valid_passes(self, pass_):
        if not pass_.isAnalysisPass:  # Analysis passes preserve all
            if pass_.ignore_preserves:
                self.valid_passes.clear()
            else:
                self.valid_passes.intersection_update(set(pass_.preserves))

        if pass_.idempotence:
            self.valid_passes.add(pass_)


class WorkingList():
    def __init__(self):
        self.list_of_items = []

    def add(self, passes, do_while=None, condition=None):

        if condition:
            self.list_of_items.append(WorkItemConditional(passes, do_while, condition))
        elif do_while:
            self.list_of_items.append(WorkItemDoWhile(passes, do_while))
        else:
            self.list_of_items.append(WorkItem(passes))

    def __iter__(self):
        for item in self.list_of_items:
            for pass_ in item:
                yield pass_


class WorkItem():
    def __init__(self, passes):
        self.passes = passes

    def __iter__(self):
        for pass_ in self.passes:
            yield pass_


class WorkItemDoWhile(WorkItem):
    def __init__(self, passes, do_while):
        self.working_list = WorkingList()
        self.working_list.add(passes)
        self.do_while = do_while

    def __iter__(self):
        while True:
            for pass_ in self.working_list:
                yield pass_
            if not self.do_while():
                break


class WorkItemConditional(WorkItem):
    def __init__(self, passes, do_while, condition):
        self.working_list = WorkingList()
        self.working_list.add(passes, do_while)
        self.condition = condition

    def __iter__(self):
        if self.condition():
            for pass_ in self.working_list:
                yield pass_
