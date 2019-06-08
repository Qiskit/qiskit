# -*- coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=attribute-defined-outside-init

"""PassContext class."""

from time import time

from .basepasses import TransformationPass, AnalysisPass


class PassContext:
    """ A wrap around the execution of a pass."""

    def __init__(self, pm_instance, pass_instance, dag):
        self.passmanager = pm_instance
        self.current_pass = pass_instance
        self.dag = dag

    def __enter__(self):
        self.enter()
        if isinstance(self.current_pass, TransformationPass):
            self.enter_transformation()
        if isinstance(self.current_pass, AnalysisPass):
            self.enter_analysis()

    def __exit__(self, *exc_info):
        self.exit(*exc_info)
        if isinstance(self.current_pass, TransformationPass):
            self.exit_transformation()
        if isinstance(self.current_pass, AnalysisPass):
            self.exit_analysis()

    def enter(self):
        pass

    def exit(self, *exc_info):
        pass

    def enter_transformation(self):
        pass

    def enter_analysis(self):
        pass

    def exit_transformation(self):
        pass

    def exit_analysis(self):
        pass


class TimeLoggerPassContext(PassContext):
    def enter(self):
        self.start_time = time()

    def exit(self, *exc_info):
        end_time = time()
        raw_log_dict = {
            'name': self.current_pass.name(),
            'start_time': self.start_time,
            'end_time': end_time,
            'running_time': end_time - self.start_time
        }
        log_dict = "%s: %.5f (ms)" % (self.passmanager.name(),
                                      (end_time - self.start_time) * 1000)
        if self.passmanager.property_set['pass_raw_log'] is None:
            self.passmanager.property_set['pass_raw_log'] = []
        if self.passmanager.property_set['pass_log'] is None:
            self.passmanager.property_set['pass_log'] = []
        self.passmanager.property_set['pass_raw_log'].append(raw_log_dict)
        self.passmanager.property_set['pass_log'].append(log_dict)
