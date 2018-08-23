# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Dummy passes used by Tranpiler testing"""

import logging

from qiskit.transpiler import TransformationPass, AnalysisPass

__all__=['DummyTP', 'DummyAP',
         'PassA_TP_NR_NP', 'PassB_TP_RA_PA', 'PassC_TP_RA_PA', 'PassD_TP_NR_NP', 'PassE_AP_NR_NP',
         'PassF_reduce_dag_property', 'PassG_calculates_dag_property', 'PassH_Bad_TP',
         'PassI_Bad_AP']

logger = "LocalLogger"

class DummyTP(TransformationPass):
    """ A dummy transformation pass."""

    def run(self, dag):
        logging.getLogger(logger).info('run transformation pass %s', self.name)


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

    def __init__(self, argument1):
        super().__init__()
        self.argument1 = argument1

    def run(self, dag):
        super().run(dag)
        logging.getLogger(logger).info('argument %s', self.argument1)


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
    """ A dummy transformation pass that (sets and)reduces a property in the DAG.
    AP: Analysis Pass
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        if not hasattr(dag, 'property'):
            dag.property = 8
        dag.property = round(dag.property * 0.8)
        logging.getLogger(logger).info('dag property = %i', dag.property)


class PassG_calculates_dag_property(DummyAP):
    """ A dummy transformation pass that (sets and)reduces a property in the DAG.
    AP: Analysis Pass
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag):
        super().run(dag)
        prop = dag.property
        self.property_set['property'] = prop
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