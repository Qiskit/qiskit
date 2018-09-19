# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,super-init-not-called

"""Dummy passes used by Tranpiler testing"""

import logging
from collections import defaultdict


from qiskit.transpiler import TransformationPass, AnalysisPass

logger = "LocalLogger"


class DummyTP(TransformationPass):
    """ A dummy transformation pass."""

    def run(self, dag):
        logging.getLogger(logger).info('run transformation pass %s', self.name)
        return dag


class DummyAP(AnalysisPass):
    """ A dummy analysis pass."""

    def run(self, dag):
        logging.getLogger(logger).info('run analysis pass %s', self.name)


class PassA_TP_NR_NP(DummyTP):
    """ A dummy pass without any requires/preserves.
    TP: Transformation Pass
    NR: No requires
    NP: No preserves
    """
    pass


class PassB_TP_RA_PA(DummyTP):
    """ A dummy pass that requires PassA_TP_NR_NP and preserves it.
    TP: Transformation Pass
    RA: Requires PassA
    PA: Preserves PassA
    """
    requires = [PassA_TP_NR_NP()]
    preserves = [PassA_TP_NR_NP()]


class PassC_TP_RA_PA(DummyTP):
    """ A dummy pass that requires PassA_TP_NR_NP and preserves it.
    TP: Transformation Pass
    RA: Requires PassA
    PA: Preserves PassA
    """
    requires = [PassA_TP_NR_NP()]
    preserves = [PassA_TP_NR_NP()]


class PassD_TP_NR_NP(DummyTP):
    """ A dummy transfomation pass that takes an argument.
    TP: Transformation Pass
    NR: No Requires
    NP: No Preserves
    """

    def __init__(self, argument1=None, argument2=None):
        self.argument1 = argument1
        self.argument2 = argument2

    def run(self, dag):
        super().run(dag)
        logging.getLogger(logger).info('argument %s', self.argument1)
        return dag


class PassE_AP_NR_NP(DummyAP):
    """ A dummy analysis pass that takes an argument.
    AP: Analysis Pass
    NR: No Requires
    NP: No Preserves
    """

    def __init__(self, argument1):
        super().__init__()
        self.argument1 = argument1

    def run(self, dag):
        super().run(dag)
        self.property_set['property'] = self.argument1
        logging.getLogger(logger).info('set property as %s', self.property_set['property'])


class PassF_reduce_dag_property(DummyTP):
    """ A dummy transformation pass that (sets and) reduces a property in the DAG.
    TP: Transformation Pass
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        if not hasattr(dag, 'property'):
            dag.property = 8
        dag.property = round(dag.property * 0.8)
        logging.getLogger(logger).info('dag property = %i', dag.property)
        return dag


class PassG_calculates_dag_property(DummyAP):
    """ A dummy transformation pass that "calculates" property in the DAG.
    AP: Analysis Pass
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        if hasattr(dag, 'property'):
            self.property_set['property'] = dag.property
        else:
            self.property_set['property'] = 8
        logging.getLogger(logger).info('set property as %s (from dag.property)',
                                       self.property_set['property'])


class PassH_Bad_TP(DummyTP):
    """ A dummy transformation pass tries to modify the property set.
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        self.property_set['property'] = "value"
        logging.getLogger(logger).info('set property as %s', self.property_set['property'])
        return dag


class PassI_Bad_AP(DummyAP):
    """ A dummy analysis pass tries to modify the dag.
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        cx_runs = dag.collect_runs(["cx"])
        logging.getLogger(logger).info('cx_runs: %s', cx_runs)
        dag._remove_op_node(cx_runs.pop()[0])
        logging.getLogger(logger).info('done removing')


class PassJ_Bad_NoReturn(DummyTP):
    """ A bad dummy transformation pass that does not return a DAG.
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        return "Something else than DAG"


class PassK_check_fixed_point(DummyAP):
    """ A dummy analysis pass that checks if a property reached a fixed point. The results is saved
    in property_set['fixed_point'][<property>] as a boolean
    AP: Analysis Pass
    NR: PassG_calculates_dag_property() when the property to check is "property"
    NP: PassG_calculates_dag_property() when the property to check is "property"
    """

    def __init__(self, property_to_check):
        self._property = property_to_check
        self._previous_value = None
        if property_to_check == "property":
            self.requires = [PassG_calculates_dag_property()]
            self.preserves = [PassG_calculates_dag_property()]

    def run(self, dag):
        super().run(dag)

        if self.property_set['fixed_point'] is None:
            self.property_set.setitem('fixed_point', defaultdict(lambda: False))
            # self.property_set['fixed_point'] = defaultdict(lambda: False)

        current_value = self.property_set[self._property]

        if self._previous_value is not None:
            self.property_set['fixed_point'][self._property] = self._previous_value == current_value

        self._previous_value = current_value

        # if self._property not in self._previous_values:
        #     self._previous_values[self._property] = current_value
        #     self.property_set['fixed_point'][self._property] = False
        # else:
        #     previous_value = self._prev_values[self._property]
        #     self.property_set['fixed_point'][self._property] = previous_value == current_value
        #     self._previous_values[self._property] = current_value
