# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Transpiler testing"""

import unittest.mock

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.transpiler import TranspilerAccessError, TranspilerError
from qiskit.transpiler.passmanager import DoWhileController, ConditionalController, \
    FlowController, FlowControllerLinear
from qiskit.test import QiskitTestCase
from ._dummy_passes import (PassA_TP_NR_NP, PassB_TP_RA_PA, PassC_TP_RA_PA,
                            PassD_TP_NR_NP, PassE_AP_NR_NP, PassF_reduce_dag_property,
                            PassH_Bad_TP, PassI_Bad_AP, PassJ_Bad_NoReturn,
                            PassK_check_fixed_point_property, PassM_AP_NR_NP)

logger = "LocalLogger"


class SchedulerTestCase(QiskitTestCase):
    """ Asserts for the scheduler. """

    def assertScheduler(self, circuit, passmanager, expected):
        """
        Runs transpiler(circuit, passmanager) and checks if the passes run as expected.
        Args:
            circuit (QuantumCircuit): Circuit to transform via transpilation.
            passmanager (PassManager): pass manager instance for the transpilation process
            expected (list): List of things the passes are logging
        """
        with self.assertLogs(logger, level='INFO') as cm:
            out = transpile(circuit, pass_manager=passmanager)
        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual([record.message for record in cm.records], expected)

    def assertSchedulerRaises(self, circuit, passmanager, expected, exception_type):
        """
        Runs transpiler(circuit, passmanager) and checks if the passes run as expected until
        exception_type is raised.
        Args:
            circuit (QuantumCircuit): Circuit to transform via transpilation
            passmanager (PassManager): pass manager instance for the transpilation process
            expected (list): List of things the passes are logging
            exception_type (Exception): Exception that is expected to be raised.
        """
        with self.assertLogs(logger, level='INFO') as cm:
            self.assertRaises(exception_type, transpile, circuit, pass_manager=passmanager)
        self.assertEqual([record.message for record in cm.records], expected)


class TestPassManagerInit(SchedulerTestCase):
    """ The pass manager sets things at init time."""

    def test_passes(self):
        """ A single chain of passes, with Requests and Preserves, at __init__ time"""
        circuit = QuantumCircuit(QuantumRegister(1))
        passmanager = PassManager(passes=[
            PassC_TP_RA_PA(),  # Request: PassA / Preserves: PassA
            PassB_TP_RA_PA(),  # Request: PassA / Preserves: PassA
            PassD_TP_NR_NP(argument1=[1, 2]),  # Requires: {}/ Preserves: {}
            PassB_TP_RA_PA()])
        self.assertScheduler(circuit, passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                    'run transformation pass PassC_TP_RA_PA',
                                                    'run transformation pass PassB_TP_RA_PA',
                                                    'run transformation pass PassD_TP_NR_NP',
                                                    'argument [1, 2]',
                                                    'run transformation pass PassA_TP_NR_NP',
                                                    'run transformation pass PassB_TP_RA_PA'])


class TestUseCases(SchedulerTestCase):
    """ The pass manager schedules passes in, sometimes, tricky ways. These tests combine passes in
     many ways, and checks that passes are ran in the right order. """

    def setUp(self):
        self.circuit = QuantumCircuit(QuantumRegister(1))
        self.passmanager = PassManager()

    def test_chain(self):
        """ A single chain of passes, with Requests and Preserves."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {}/ Preserves: {}
        self.passmanager.append(PassB_TP_RA_PA())
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassC_TP_RA_PA',
                              'run transformation pass PassB_TP_RA_PA',
                              'run transformation pass PassD_TP_NR_NP',
                              'argument [1, 2]',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassB_TP_RA_PA'])

    def test_conditional_passes_true(self):
        """ A pass set with a conditional parameter. The callable is True. """
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(PassA_TP_NR_NP(),
                                condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run analysis pass PassE_AP_NR_NP',
                              'set property as True',
                              'run transformation pass PassA_TP_NR_NP'])

    def test_conditional_passes_false(self):
        """ A pass set with a conditional parameter. The callable is False. """
        self.passmanager.append(PassE_AP_NR_NP(False))
        self.passmanager.append(PassA_TP_NR_NP(),
                                condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassE_AP_NR_NP',
                                                              'set property as False'])

    def test_conditional_and_loop(self):
        """ Run a conditional first, then a loop"""
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'],
            condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run analysis pass PassE_AP_NR_NP',
                              'set property as True',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 8 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 6',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 6 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 5',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 5 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 4',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 4 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 3',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 3 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2'])

    def test_loop_and_conditional(self):
        """ Run a loop first, then a conditional"""
        FlowController.remove_flow_controller('condition')
        FlowController.add_flow_controller('condition', ConditionalController)

        self.passmanager.append(PassK_check_fixed_point_property())
        self.passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'],
            condition=lambda property_set: not property_set['property_fixed_point'])
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run analysis pass PassG_calculates_dag_property',
                              'set property as 8 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 6',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 6 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 5',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 5 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 4',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 4 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 3',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 3 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2'])

    def test_do_not_repeat_based_on_preservation(self):
        """ When a pass is still a valid pass (because following passes preserved it), it should not
        run again"""
        self.passmanager.append([PassB_TP_RA_PA(), PassA_TP_NR_NP(), PassB_TP_RA_PA()])
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassB_TP_RA_PA'])

    def test_do_not_repeat_based_on_idempotence(self):
        """ Repetition can be optimized to a single execution when the pass is idempotent"""
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append([PassA_TP_NR_NP(), PassA_TP_NR_NP()])
        self.passmanager.append(PassA_TP_NR_NP())
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run transformation pass PassA_TP_NR_NP'])

    def test_non_idempotent_pass(self):
        """ Two or more runs of a non-idempotent pass cannot be optimized. """
        self.passmanager.append(PassF_reduce_dag_property())
        self.passmanager.append([PassF_reduce_dag_property(), PassF_reduce_dag_property()])
        self.passmanager.append(PassF_reduce_dag_property())
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run transformation pass PassF_reduce_dag_property',
                              'dag property = 6',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 5',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 4',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 3'])

    def test_fenced_property_set(self):
        """ Transformation passes are not allowed to modified the property set. """
        self.passmanager.append(PassH_Bad_TP())
        self.assertSchedulerRaises(self.circuit, self.passmanager,
                                   ['run transformation pass PassH_Bad_TP'],
                                   TranspilerAccessError)

    def test_fenced_dag(self):
        """ Analysis passes are not allowed to modified the DAG. """
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.cx(qr[0], qr[1])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[1], qr[0])
        circ.cx(qr[1], qr[0])

        self.passmanager.append(PassI_Bad_AP())
        self.assertSchedulerRaises(circ, self.passmanager,
                                   ['run analysis pass PassI_Bad_AP',
                                    'cx_runs: {(5, 6, 7, 8)}'],
                                   TranspilerAccessError)

    def test_ignore_request_pm(self):
        """ A pass manager that ignores requires does not run the passes decleared in the 'requires'
        field of the passes."""
        passmanager = PassManager(ignore_requires=True)
        passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.append(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {} / Preserves: {}
        passmanager.append(PassB_TP_RA_PA())
        self.assertScheduler(self.circuit, passmanager, ['run transformation pass PassC_TP_RA_PA',
                                                         'run transformation pass PassB_TP_RA_PA',
                                                         'run transformation pass PassD_TP_NR_NP',
                                                         'argument [1, 2]',
                                                         'run transformation pass PassB_TP_RA_PA'])

    def test_ignore_preserves_pm(self):
        """ A pass manager that ignores preserves does not record the passes decleared in the
        'preserves' field of the passes as valid passes."""
        passmanager = PassManager(ignore_preserves=True)
        passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA
        passmanager.append(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {} / Preserves: {}
        passmanager.append(PassB_TP_RA_PA())
        self.assertScheduler(self.circuit, passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassC_TP_RA_PA',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassB_TP_RA_PA',
                                                         'run transformation pass PassD_TP_NR_NP',
                                                         'argument [1, 2]',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassB_TP_RA_PA'])

    def test_pass_non_idempotence_pm(self):
        """ A pass manager that considers every pass as not idempotent, allows the immediate
        repetition of a pass"""
        passmanager = PassManager(ignore_preserves=True)
        passmanager.append(PassA_TP_NR_NP())
        passmanager.append(PassA_TP_NR_NP())  # Normally removed for optimization, not here.
        passmanager.append(PassB_TP_RA_PA())  # Normally required is ignored for optimization,
        # not here
        self.assertScheduler(self.circuit, passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassB_TP_RA_PA'])

    def test_pass_non_idempotence_passset(self):
        """ A pass set that is not idempotent. """
        passmanager = PassManager()
        passmanager.append([PassA_TP_NR_NP(), PassB_TP_RA_PA()], ignore_preserves=True)
        self.assertScheduler(self.circuit, passmanager, ['run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run transformation pass PassB_TP_RA_PA'])

    def test_analysis_pass_is_idempotent(self):
        """ Analysis passes are idempotent. """
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, passmanager, ['run analysis pass PassE_AP_NR_NP',
                                                         'set property as 1'])

    def test_ap_before_and_after_a_tp(self):
        """ A default transformation does not preserves anything and analysis passes
        need to be re-run"""
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassA_TP_NR_NP())
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, passmanager, ['run analysis pass PassE_AP_NR_NP',
                                                         'set property as 1',
                                                         'run transformation pass PassA_TP_NR_NP',
                                                         'run analysis pass PassE_AP_NR_NP',
                                                         'set property as 1'])

    def test_pass_option_precedence(self):
        """ The precedence of options is, in order of priority:
         - The passset option
         - The Pass Manager option
         - Default
        """
        passmanager = PassManager(ignore_preserves=False, ignore_requires=True)
        tp_pass = PassA_TP_NR_NP()
        passmanager.append(tp_pass, ignore_preserves=True)
        the_pass_in_the_workinglist = next(iter(passmanager.working_list))
        self.assertTrue(the_pass_in_the_workinglist.options['ignore_preserves'])
        self.assertTrue(the_pass_in_the_workinglist.options['ignore_requires'])

    def test_pass_no_return_a_dag(self):
        """ Passes instances with same arguments (independently of the order) are the same. """
        self.passmanager.append(PassJ_Bad_NoReturn())
        self.assertSchedulerRaises(self.circuit, self.passmanager,
                                   ['run transformation pass PassJ_Bad_NoReturn'], TranspilerError)

    def test_fixed_point_pass(self):
        """ A pass set with a do_while parameter that checks for a fixed point. """
        self.passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'])
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run analysis pass PassG_calculates_dag_property',
                              'set property as 8 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 6',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 6 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 5',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 5 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 4',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 4 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 3',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 3 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2',
                              'run analysis pass PassG_calculates_dag_property',
                              'set property as 2 (from dag.property)',
                              'run analysis pass PassK_check_fixed_point_property',
                              'run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassF_reduce_dag_property',
                              'dag property = 2'])

    def test_fixed_point_pass_max_iteration(self):
        """ A pass set with a do_while parameter that checks that the max_iteration is raised. """
        self.passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'],
            max_iteration=2)
        self.assertSchedulerRaises(self.circuit, self.passmanager,
                                   ['run analysis pass PassG_calculates_dag_property',
                                    'set property as 8 (from dag.property)',
                                    'run analysis pass PassK_check_fixed_point_property',
                                    'run transformation pass PassA_TP_NR_NP',
                                    'run transformation pass PassF_reduce_dag_property',
                                    'dag property = 6',
                                    'run analysis pass PassG_calculates_dag_property',
                                    'set property as 6 (from dag.property)',
                                    'run analysis pass PassK_check_fixed_point_property',
                                    'run transformation pass PassA_TP_NR_NP',
                                    'run transformation pass PassF_reduce_dag_property',
                                    'dag property = 5'], TranspilerError)

    def test_fresh_initial_state(self):
        """ New construction gives fresh instance """
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run analysis pass PassM_AP_NR_NP',
                              'self.argument1 = 2',
                              'run transformation pass PassA_TP_NR_NP',
                              'run analysis pass PassM_AP_NR_NP',
                              'self.argument1 = 2'])


class DoXTimesController(FlowController):
    """ A control-flow plugin for running a set of passes an X amount of times."""

    def __init__(self, passes, options, do_x_times=0, **_):  # pylint: disable=super-init-not-called
        self.do_x_times = do_x_times()
        super().__init__(passes, options)

    def __iter__(self):
        for _ in range(self.do_x_times):
            for pass_ in self.passes:
                yield pass_


class TestControlFlowPlugin(SchedulerTestCase):
    """ Testing the control flow plugin system. """

    def setUp(self):
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_control_flow_plugin(self):
        """ Adds a control flow plugin with a single parameter and runs it. """
        FlowController.add_flow_controller('do_x_times', DoXTimesController)
        self.passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_x_times=lambda x: 3)
        self.assertScheduler(self.circuit, self.passmanager,
                             ['run transformation pass PassA_TP_NR_NP',
                              'run transformation pass PassB_TP_RA_PA',
                              'run transformation pass PassC_TP_RA_PA',
                              'run transformation pass PassB_TP_RA_PA',
                              'run transformation pass PassC_TP_RA_PA',
                              'run transformation pass PassB_TP_RA_PA',
                              'run transformation pass PassC_TP_RA_PA'])

    def test_callable_control_flow_plugin(self):
        """ Removes do_while, then adds it back. Checks max_iteration still working. """
        controllers_length = len(FlowController.registered_controllers)
        FlowController.remove_flow_controller('do_while')
        self.assertEqual(controllers_length - 1, len(FlowController.registered_controllers))
        FlowController.add_flow_controller('do_while', DoWhileController)
        self.assertEqual(controllers_length, len(FlowController.registered_controllers))
        self.passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()],
                                do_while=lambda property_set: True, max_iteration=2)
        self.assertSchedulerRaises(self.circuit, self.passmanager,
                                   ['run transformation pass PassA_TP_NR_NP',
                                    'run transformation pass PassB_TP_RA_PA',
                                    'run transformation pass PassC_TP_RA_PA',
                                    'run transformation pass PassB_TP_RA_PA',
                                    'run transformation pass PassC_TP_RA_PA'], TranspilerError)

    def test_remove_nonexistent_plugin(self):
        """ Tries to remove a plugin that does not exist. """
        self.assertRaises(KeyError, FlowController.remove_flow_controller, "foo")


class TestDumpPasses(SchedulerTestCase):
    """ Testing the passes method. """

    def test_passes(self):
        """Dump passes in different FlowControllerLinear"""
        passmanager = PassManager()
        passmanager.append(PassC_TP_RA_PA())
        passmanager.append(PassB_TP_RA_PA())

        expected = [{'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassC_TP_RA_PA()],
                     'type': FlowControllerLinear},
                    {'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassB_TP_RA_PA()],
                     'type': FlowControllerLinear}]
        self.assertEqual(expected, passmanager.passes())

    def test_passes_in_linear(self):
        """Dump passes in the same FlowControllerLinear"""
        passmanager = PassManager(passes=[
            PassC_TP_RA_PA(),
            PassB_TP_RA_PA(),
            PassD_TP_NR_NP(argument1=[1, 2]),
            PassB_TP_RA_PA()])

        expected = [{'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassC_TP_RA_PA(),
                                PassB_TP_RA_PA(),
                                PassD_TP_NR_NP(argument1=[1, 2]),
                                PassB_TP_RA_PA()],
                     'type': FlowControllerLinear}]
        self.assertEqual(expected, passmanager.passes())

    def test_control_flow_plugin(self):
        """ Dump passes in a custom flow controller. """
        passmanager = PassManager()
        FlowController.add_flow_controller('do_x_times', DoXTimesController)
        passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_x_times=lambda x: 3)

        expected = [{'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassB_TP_RA_PA(),
                                PassC_TP_RA_PA()],
                     'type': DoXTimesController}]
        self.assertEqual(expected, passmanager.passes())

    def test_conditional_and_loop(self):
        """ Dump passes with a conditional and a loop"""
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(True))
        passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'],
            condition=lambda property_set: property_set['property_fixed_point'])

        expected = [{'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassE_AP_NR_NP(True)],
                     'type': FlowControllerLinear},
                    {'options': {'ignore_preserves': False,
                                 'ignore_requires': False,
                                 'max_iteration': 1000},
                     'passes': [PassK_check_fixed_point_property(),
                                PassA_TP_NR_NP(),
                                PassF_reduce_dag_property()],
                     'type': ConditionalController}]
        self.assertEqual(expected, passmanager.passes())


class TestPassManagerReuse(SchedulerTestCase):
    """ The PassManager instance should be resusable"""
    def setUp(self):
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_chain_twice(self):
        """ Run a chain twice."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        expected = ['run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassC_TP_RA_PA',
                    'run transformation pass PassB_TP_RA_PA']

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_conditional_twice(self):
        """ Run a conditional twice. """
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(PassA_TP_NR_NP(),
                                condition=lambda property_set: property_set['property'])

        expected = ['run analysis pass PassE_AP_NR_NP',
                    'set property as True',
                    'run transformation pass PassA_TP_NR_NP']

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_fixed_point_twice(self):
        """ A fixed point scheduler, twice """
        self.passmanager.append(
            [PassK_check_fixed_point_property(),
             PassA_TP_NR_NP(),
             PassF_reduce_dag_property()],
            do_while=lambda property_set: not property_set['property_fixed_point'])

        expected = ['run analysis pass PassG_calculates_dag_property',
                    'set property as 8 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 6',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 6 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 5',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 5 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 4',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 4 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 3',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 3 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 2',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 2 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 2',
                    'run analysis pass PassG_calculates_dag_property',
                    'set property as 2 (from dag.property)',
                    'run analysis pass PassK_check_fixed_point_property',
                    'run transformation pass PassA_TP_NR_NP',
                    'run transformation pass PassF_reduce_dag_property',
                    'dag property = 2']

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)


if __name__ == '__main__':
    unittest.main()
