# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================
Passmanager (:mod:`qiskit.passmanager`)
=======================================

.. currentmodule:: qiskit.passmanager

Overview
========

Qiskit pass manager is somewhat inspired by the `LLVM compiler <https://llvm.org/>`_,
but it is designed to take Python object as an input instead of plain source code.

The pass manager converts the input Python object into an intermediate representation (IR),
and it can be optimized and get lowered with a variety of transformations over multiple passes.
The pass manager framework may employ multiple IRs with interleaved conversion passes,
depending on the context of the optimization.

.. note::

    Currently there is no actual use/design of multiple IRs in builtin Qiskit pass managers.
    At lest implementation of the :mod:`~.passmanager` module is agnostic to
    actual IR types (i.e. no strict type check is performed), and the pass manager works
    as long as the IR implements all methods required by subsequent passes.
    Concrete design for the use of multiple IR might be provided in the future release.

The passes may consume the hardware constraints that Qiskit backend may provide.
Finally, the IR is converted back to some Python object.
Note that the input type and output type are not necessary consistent.

Compilation in the pass manager is a chain of :class:`~.passmanager.Task` execution that
takes an IR and outputs new IR with some optimization or data analysis.
An atomic task is a `pass` which is a subclass of  the :class:`.GenericPass` that implements
a :meth:`.~GenericPass.run` method that performs some work on the received IR.
A set of passes may form a `flow controller` which is a subclass of the
:class:`.BaseFlowController`, that may repeatedly execute registered passes
based on user provided conditions.
Passes can share intermediate data via the :class:`.PropertySet` object which is
a free form dictionary. A pass can populate the property set dictionary during the task execution.
A flow controller can also consume the property set to control the pass execution,
but this access must be read-only.
The property set is portable and handed over from pass to pass at execution.
In addition to the property set, task can also receive :class:`.PassState` data structure.
This object is updated once after every pass logic is run, and thus the lifetime of the
data is only during execution of a single pass.
Pass or flow controller may determine termination condition based on the pass status.


A pass manager is a wrapper of the flow controller, with responsibilities of

* Scheduling optimization tasks,
* Converting an input Python object to a particular Qiskit IR,
* Initializing a property set and pass state,
* Running scheduled tasks to apply a series of transformations to the IR,
* Converting the IR back to an output Python object.

This indicates that the flow controller itself is type-agnostic, and a developer must
implement a subclass of the :class:`BasePassManager` to manage the data conversion steps.
This `veil of ignorance` allows us to choose the most efficient data representation
for a particular pass manager task, while we can reuse the flow control machinery
for different input and output types.

A single flow controller always takes a single IR object, and returns a single
IR object. Parallelism for multiple input objects is supported by the
:class:`BasePassManager` by broadcasting the flow controller via
the :mod:`qiskit.tools.parallel_map` function.


Examples:

    We look into a toy optimization task, namely, preparing a row of numbers
    and remove a digit if the number is five.
    Such task might be easily done by converting the input numbers into string.
    We dare to use the pass manager framework here, putting the efficiency aside for
    a moment to learn how to build a custom Qiskit compiler.

    .. code-block:: python

        from qiskit.passmanager import BasePassManager, GenericPass, ConditionalController

        class ToyPassManager(BasePassManager):

            def _passmanager_frontend(self, input_program: int, **kwargs) -> str:
                return str(input_program)

            def _passmanager_backend(self, passmanager_ir: str, **kwargs) -> int:
                return int(passmanager_ir)

    This pass manager inputs and outputs an integer number, while
    performing the optimization tasks on a string data.
    Hence, input, IR, output type are integer, string, integer, respectively.
    The :meth:`.~BasePassManager._passmanager_frontend` defines a conversion from the
    input data to IR, and :meth:`.~BasePassManager._passmanager_backend` defines
    a conversion from the IR to output data.

    Next, we implement a pass that removes a digit when the number is five.

    .. code-block:: python

        class RemoveFive(GenericPass):

            def run(self, passmanager_ir: str):
                return passmanager_ir.replace("5", "")

        task = RemoveFive()

    Finally, we instantiate a pass manager and schedule the task with it.
    Running the pass manager with random row of numbers returns
    new numbers that don't contain five.

    .. code-block:: python

        pm = ToyPassManager()
        pm.append(task)

        pm.run([123456789, 45654, 36785554])

    Output:

    .. parsed-literal::

        [12346789, 464, 36784]

    Now we consider the case of conditional execution.
    We avoid execution of the remove five task when the input number is
    six digit or less. Such control can be implemented by a flow controller.
    We start from an analysis pass that provides the flow controller
    with information about the number of digits.

    .. code-block:: python

        class CountDigits(GenericPass):

            def run(self, passmanager_ir: str):
                self.property_set["ndigits"] = len(passmanager_ir)

        analysis_task = CountDigits()

    Then, we wrap the remove five task with the :class:`.ConditionalController`
    that runs the stored tasks only when the condition is met.

    .. code-block:: python

        def digit_condition(property_set):
            # Return True when condition is met.
            return property_set["ndigits"] > 6

        conditional_task = ConditionalController(
            passes=[RemoveFive()],
            condition=digit_condition,
        )

    As before, we schedule these passes with the pass manager and run.

    .. code-block:: python

        pm = ToyPassManager()
        pm.append(analysis_task)
        pm.append(conditional_task)

        pm.run([123456789, 45654, 36785554])

    Output:

    .. parsed-literal::

        [12346789, 45654, 36784]

    The remove five task is triggered only for the first and third input
    values with more than six digits.

    With the pass manager framework, a developer can flexibly customize
    the optimization task by combining multiple passes and flow controllers.
    See details for following class API documentations.


Base classes
------------

.. autosummary::
   :toctree: ../stubs/

   BasePassManager
   BaseFlowController
   GenericPass

Flow controllers
----------------

.. autosummary::
   :toctree: ../stubs/

   FlowController
   ConditionalController
   DoWhileController

PropertySet
-----------

.. autosummary::
   :toctree: ../stubs/

   PropertySet
   FencedPropertySet
   PassState

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   PassManagerError

"""

from .passmanager import BasePassManager
from .flow_controllers import FlowController, ConditionalController, DoWhileController
from .base_tasks import GenericPass, BaseFlowController
from .propertyset import PropertySet, FencedPropertySet, PassState
from .exceptions import PassManagerError
