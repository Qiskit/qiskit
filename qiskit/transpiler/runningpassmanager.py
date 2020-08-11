# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RunningPassManager class for the transpiler.
This object holds the state of a pass manager during running-time."""

from functools import partial
from collections import OrderedDict
import logging
from time import time

from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from .propertyset import PropertySet
from .fencedobjs import FencedPropertySet, FencedDAGCircuit
from .exceptions import TranspilerError

logger = logging.getLogger(__name__)


class RunningPassManager:
    """A RunningPassManager is a running pass manager."""

    def __init__(self, max_iteration):
        """Initialize an empty PassManager object (with no passes scheduled).

        Args:
            max_iteration (int): The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        self.callback = None
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().
        self.working_list = []

        # global property set is the context of the circuit held by the pass manager
        # as it runs through its scheduled passes. The flow controller
        # have read-only access (via the fenced_property_set).
        self.property_set = PropertySet()
        self.fenced_property_set = FencedPropertySet(self.property_set)

        # passes already run that have not been invalidated
        self.valid_passes = set()

        # pass manager's overriding options for the passes it runs (for debugging)
        self.passmanager_options = {'max_iteration': max_iteration}

        self.count = 0

    def append(self, passes, **flow_controller_conditions):
        """Append a Pass to the schedule of passes.

        Args:
            passes (list[BasePass]): passes to be added to schedule
            flow_controller_conditions (kwargs): See add_flow_controller(): Dictionary of
            control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
        """
        flow_controller_conditions = self._normalize_flow_controller(flow_controller_conditions)

        self.working_list.append(
            FlowController.controller_factory(passes,
                                              self.passmanager_options,
                                              **flow_controller_conditions))

    def _normalize_flow_controller(self, flow_controller):
        for name, param in flow_controller.items():
            if callable(param):
                flow_controller[name] = partial(param, self.fenced_property_set)
            else:
                raise TranspilerError('The flow controller parameter %s is not callable' % name)
        return flow_controller

    def run(self, circuit, output_name=None, callback=None):
        """Run all the passes on a QuantumCircuit

        Args:
            circuit (QuantumCircuit): circuit to transform via all the registered passes
            output_name (str): The output circuit name. If not given, the same as the
                               input circuit
            callback (callable): A callback function that will be called after each pass execution.
        Returns:
            QuantumCircuit: Transformed circuit.
        """
        name = circuit.name
        dag = circuit_to_dag(circuit)
        del circuit

        if callback:
            self.callback = callback

        for passset in self.working_list:
            for pass_ in passset:
                dag = self._do_pass(pass_, dag, passset.options)

        circuit = dag_to_circuit(dag)
        if output_name:
            circuit.name = output_name
        else:
            circuit.name = name
        circuit._layout = self.property_set['layout']
        return circuit

    def _do_pass(self, pass_, dag, options):
        """Do a pass and its "requires".

        Args:
            pass_ (BasePass): Pass to do.
            dag (DAGCircuit): The dag on which the pass is ran.
            options (dict): PassManager options.
        Returns:
            DAGCircuit: The transformed dag in case of a transformation pass.
            The same input dag in case of an analysis pass.
        Raises:
            TranspilerError: If the pass is not a proper pass instance.
        """

        # First, do the requires of pass_
        for required_pass in pass_.requires:
            dag = self._do_pass(required_pass, dag, options)

        # Run the pass itself, if not already run
        if pass_ not in self.valid_passes:
            dag = self._run_this_pass(pass_, dag)

            # update the valid_passes property
            self._update_valid_passes(pass_)

        return dag

    def _run_this_pass(self, pass_, dag):
        pass_.property_set = self.property_set
        if pass_.is_transformation_pass:
            # Measure time if we have a callback or logging set
            start_time = time()
            new_dag = pass_.run(dag)
            end_time = time()
            run_time = end_time - start_time
            # Execute the callback function if one is set
            if self.callback:
                self.callback(pass_=pass_, dag=new_dag,
                              time=run_time,
                              property_set=self.property_set,
                              count=self.count)
                self.count += 1
            self._log_pass(start_time, end_time, pass_.name())
            if not isinstance(new_dag, DAGCircuit):
                raise TranspilerError("Transformation passes should return a transformed dag."
                                      "The pass %s is returning a %s" % (type(pass_).__name__,
                                                                         type(new_dag)))
            dag = new_dag
        elif pass_.is_analysis_pass:
            # Measure time if we have a callback or logging set
            start_time = time()
            pass_.run(FencedDAGCircuit(dag))
            end_time = time()
            run_time = end_time - start_time
            # Execute the callback function if one is set
            if self.callback:
                self.callback(pass_=pass_, dag=dag,
                              time=run_time,
                              property_set=self.property_set,
                              count=self.count)
                self.count += 1
            self._log_pass(start_time, end_time, pass_.name())
        else:
            raise TranspilerError("I dont know how to handle this type of pass")
        return dag

    def _log_pass(self, start_time, end_time, name):
        log_msg = "Pass: %s - %.5f (ms)" % (
            name, (end_time - start_time) * 1000)
        logger.info(log_msg)

    def _update_valid_passes(self, pass_):
        self.valid_passes.add(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            self.valid_passes.intersection_update(set(pass_.preserves))


class FlowController():
    """Base class for multiple types of working list.

    This class is a base class for multiple types of working list. When you iterate on it, it
    returns the next pass to run.
    """

    registered_controllers = OrderedDict()

    def __init__(self, passes, options, **partial_controller):
        self._passes = passes
        self.passes = FlowController.controller_factory(passes, options, **partial_controller)
        self.options = options

    def __iter__(self):
        for pass_ in self.passes:
            yield pass_

    def dump_passes(self):
        """Fetches the passes added to this flow controller.

        Returns:
             dict: {'options': self.options, 'passes': [passes], 'type': type(self)}
        """
        # TODO remove
        ret = {'options': self.options, 'passes': [], 'type': type(self)}
        for pass_ in self._passes:
            if isinstance(pass_, FlowController):
                ret['passes'].append(pass_.dump_passes())
            else:
                ret['passes'].append(pass_)
        return ret

    @classmethod
    def add_flow_controller(cls, name, controller):
        """Adds a flow controller.

        Args:
            name (string): Name of the controller to add.
            controller (type(FlowController)): The class implementing a flow controller.
        """
        cls.registered_controllers[name] = controller

    @classmethod
    def remove_flow_controller(cls, name):
        """Removes a flow controller.

        Args:
            name (string): Name of the controller to remove.
        Raises:
            KeyError: If the controller to remove was not registered.
        """
        if name not in cls.registered_controllers:
            raise KeyError("Flow controller not found: %s" % name)
        del cls.registered_controllers[name]

    @classmethod
    def controller_factory(cls, passes, options, **partial_controller):
        """Constructs a flow controller based on the partially evaluated controller arguments.

        Args:
            passes (list[BasePass]): passes to add to the flow controller.
            options (dict): PassManager options.
            **partial_controller (dict): Partially evaluated controller arguments in the form
                `{name:partial}`

        Raises:
            TranspilerError: When partial_controller is not well-formed.

        Returns:
            FlowController: A FlowController instance.
        """
        if None in partial_controller.values():
            raise TranspilerError('The controller needs a condition.')

        if partial_controller:
            for registered_controller in cls.registered_controllers.keys():
                if registered_controller in partial_controller:
                    return cls.registered_controllers[registered_controller](passes, options,
                                                                             **partial_controller)
            raise TranspilerError("The controllers for %s are not registered" % partial_controller)

        return FlowControllerLinear(passes, options)


class FlowControllerLinear(FlowController):
    """The basic controller runs the passes one after the other."""

    def __init__(self, passes, options):  # pylint: disable=super-init-not-called
        self.passes = self._passes = passes
        self.options = options


class DoWhileController(FlowController):
    """Implements a set of passes in a do-while loop."""

    def __init__(self, passes, options, do_while=None,
                 **partial_controller):
        self.do_while = do_while
        self.max_iteration = options['max_iteration']
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        for _ in range(self.max_iteration):
            for pass_ in self.passes:
                yield pass_

            if not self.do_while():
                return

        raise TranspilerError("Maximum iteration reached. max_iteration=%i" % self.max_iteration)


class ConditionalController(FlowController):
    """Implements a set of passes under a certain condition."""

    def __init__(self, passes, options, condition=None,
                 **partial_controller):
        self.condition = condition
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        if self.condition():
            for pass_ in self.passes:
                yield pass_


# Default controllers
FlowController.add_flow_controller('condition', ConditionalController)
FlowController.add_flow_controller('do_while', DoWhileController)
