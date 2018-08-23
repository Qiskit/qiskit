# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Tranpiler testing"""

import unittest.mock
import logging

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager, transpile, TransformationPass, AnalysisPass, \
    TranspilerAccessError
from ..common import QiskitTestCase

logger = "LocalLogger"


class DummyTP(TransformationPass):
    """ A dummy transformation pass."""

    def run(self, dag, property_set):
        logging.getLogger(logger).info('run transformation pass %s', self.name)


class DummyAP(AnalysisPass):
    """ A dummy analysis pass."""

    def run(self, dag, property_set):
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

    def run(self, dag, property_set):
        super().run(dag, property_set)
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

    def run(self, dag, property_set):
        super().run(dag, property_set)
        property_set['property'] = self.argument1
        logging.getLogger(logger).info('set property as %s', property_set['property'])


class PassF_reduce_dag_property(DummyTP):
    """ A dummy transformation pass that (sets and)reduces a property in the DAG.
    AP: Analysis Pass
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag, property_set):
        super().run(dag, property_set)
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

    def run(self, dag, property_set):
        super().run(dag, property_set)
        prop = dag.property
        property_set['property'] = prop
        logging.getLogger(logger).info('set property as %s (from dag.property)',
                                       property_set['property'])


class PassH_Bad_TP(DummyTP):
    """ A dummy transformation pass tries to modify the property set.
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag, property_set):
        super().run(dag, property_set)
        property_set['property'] = "value"
        logging.getLogger(logger).info('set property as %s', property_set['property'])


class PassI_Bad_AP(DummyAP):
    """ A dummy analysis pass tries to modify the dag.
    NR: No Requires
    NP: No Preserves
    """

    def run(self, dag, property_set):
        super().run(dag, property_set)
        cx_runs = dag.collect_runs(["cx"])
        logging.getLogger(logger).info('cx_runs: %s', cx_runs)
        dag._remove_op_node(cx_runs.pop()[0])
        logging.getLogger(logger).info('done removing')


class TestUseCases(QiskitTestCase):
    """ The pass manager schedules passes in, sometimes, tricky ways. These tests combine passes in
     many ways, and checks that passes are ran in the right order. """

    def setUp(self):
        self.dag = DAGCircuit.fromQuantumCircuit(QuantumCircuit(QuantumRegister(1)))
        self.passmanager = PassManager()

    def assertScheduler(self, dag, passmanager, expected):
        """
        Runs transpiler(dag, passmanager) and checks if the passes run as expected.
        Args:
            dag (DAGCircuit): DAG circuit to transform via transpilation
            passmanager (PassManager): pass manager instance for the tranpilation process
            expected (list): List of things the passes are logging
        """
        with self.assertLogs(logger, level='INFO') as cm:
            transpile(dag, pass_manager=passmanager)
        self.assertEqual([record.message for record in cm.records], expected)

    def assertSchedulerRaises(self, dag, passmanager, expected, exception_type):
        """
        Runs transpiler(dag, passmanager) and checks if the passes run as expected until
        expcetion_type is raised.
        Args:
            dag (DAGCircuit): DAG circuit to transform via transpilation
            passmanager (PassManager): pass manager instance for the tranpilation process
            expected (list): List of things the passes are logging
            exception_type (Exception): Exception that is expected to be raised.
        """
        with self.assertLogs(logger, level='INFO') as cm:
            self.assertRaises(exception_type, transpile, dag, pass_manager=passmanager)
        self.assertEqual([record.message for record in cm.records], expected)

    def test_chain(self):
        """ A single chain of passes, with Requests and Preserves."""
        self.passmanager.add_pass(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.add_pass(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.add_pass(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {} / Preserves: {}
        self.passmanager.add_pass(PassB_TP_RA_PA())
        self.assertScheduler(self.dag, self.passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                          'run transformation pass PassC_TP_RA_PA',
                                                          'run transformation pass PassB_TP_RA_PA',
                                                          'run transformation pass PassD_TP_NR_NP',
                                                          'argument [1, 2]',
                                                          'run transformation pass PassA_TP_NR_NP',
                                                          'run transformation pass PassB_TP_RA_PA'])

    def test_conditional_passes_true(self):
        """ A pass set with a conditional parameter. The callable is True. """
        self.passmanager.add_pass(PassE_AP_NR_NP(True))
        self.passmanager.add_pass(PassA_TP_NR_NP(),
                                  condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.dag, self.passmanager, ['run analysis pass PassE_AP_NR_NP',
                                                          'set property as True',
                                                          'run transformation pass PassA_TP_NR_NP'])

    def test_conditional_passes_false(self):
        """ A pass set with a conditional parameter. The callable is False. """
        self.passmanager.add_pass(PassE_AP_NR_NP(False))
        self.passmanager.add_pass(PassA_TP_NR_NP(),
                                  condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.dag, self.passmanager, ['run analysis pass PassE_AP_NR_NP',
                                                          'set property as False'])

    def test_do_while_until_fixed_point(self):
        self.passmanager.add_pass([
            PassF_reduce_dag_property(),
            PassA_TP_NR_NP(),  # Since preserves nothings,  allows PassF to loop
            PassG_calculates_dag_property()], \
            do_while=lambda property_set: not property_set.fixed_point('property'))
        self.assertScheduler(self.dag, self.passmanager,
                             ['run transformation pass PassF_reduce_dag_property',
                              'dag property = 6',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 6 (from dag.property)',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 5',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 5 (from dag.property)',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 4',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 4 (from dag.property)',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 3',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 3 (from dag.property)',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)'])

    def test_do_not_repeat_based_on_preservation(self):
        """ When a pass is still a valid pass (because following passes preserved it), it should not
        run again"""
        self.passmanager.add_pass([PassB_TP_RA_PA(), PassA_TP_NR_NP(), PassB_TP_RA_PA()])
        self.assertScheduler(self.dag, self.passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                          'run transformation pass PassB_TP_RA_PA'])

    def test_do_not_repeat_based_on_idempotence(self):
        """ By default, passes are idempotent. Therefore, repetition can be optimized to a single
        execution"""
        self.passmanager.add_pass(PassA_TP_NR_NP())
        self.passmanager.add_pass([PassA_TP_NR_NP(), PassA_TP_NR_NP()])
        self.passmanager.add_pass(PassA_TP_NR_NP())
        self.assertScheduler(self.dag, self.passmanager, ['run transformation pass PassA_TP_NR_NP'])

    def test_fenced_property_set(self):
        self.passmanager.add_pass(PassH_Bad_TP())
        self.assertSchedulerRaises(self.dag, self.passmanager,
                                   ['run transformation pass PassH_Bad_TP'],
                                   TranspilerAccessError)

    def test_fenced_dag(self):
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.cx(qr[0], qr[1])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[1], qr[0])
        circ.cx(qr[1], qr[0])
        dag = DAGCircuit.fromQuantumCircuit(circ)

        self.passmanager.add_pass(PassI_Bad_AP())
        self.assertSchedulerRaises(dag, self.passmanager,
                                   ['run analysis pass PassI_Bad_AP',
                                    'cx_runs: {(5, 6, 7, 8)}'],
                                   TranspilerAccessError)

    def test_ignore_request_pm(self):
        """ A single chain of passes, with Requests and Preserves."""
        passmanager = PassManager(ignore_requests=True)
        passmanager.add_pass(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.add_pass(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.add_pass(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {} / Preserves: {}
        passmanager.add_pass(PassB_TP_RA_PA())
        self.assertScheduler(self.dag, passmanager, ['run transformation pass PassC_TP_RA_PA',
                                                     'run transformation pass PassB_TP_RA_PA',
                                                     'run transformation pass PassD_TP_NR_NP',
                                                     'argument [1, 2]',
                                                     'run transformation pass PassB_TP_RA_PA'])


if __name__ == '__main__':
    unittest.main()
