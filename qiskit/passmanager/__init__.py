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
but it is designed to take Qiskit object as an input instead of plain source code.

The pass manager converts the input object into an intermediate representation (IR),
and it can be optimized and get lowered with a variety of transformations over multiple passes.
This representation must be preserved throughout the transformation.
The passes may consume the hardware constraints that Qiskit backend may provide.
Finally, the IR is converted back to some Qiskit object.
Note that the input type and output type don't need to match.

Execution of passes is managed by the :class:`.FlowController`,
which is initialized with a set of transform and analysis passes and provides an iterator of them.
This iterator can be conditioned on the :class:`.PropertySet`, which is a namespace
storing the intermediate data necessary for the transformation.
A pass has read and write access to the property set namespace,
and the stored data is shared among scheduled passes.

The :class:`BasePassManager` provides a user interface to build and execute transform passes.
It internally spawns a :class:`BasePassRunner` instance to apply transforms to
the input object. In this sense, the pass manager itself is unaware of the
underlying IR, but it is indirectly tied to a particular IR through the pass runner class.

The responsibilities of the pass runner are the following:

* Defining the input type / pass manager IR / output type.
* Converting an input object to a particular pass manager IR.
* Preparing own property set namespace.
* Running scheduled flow controllers to apply a series of transformations to the IR.
* Converting the IR back to an output object.

A single pass runner always takes a single input object and returns a single output object.
Parallelism for multiple input objects is supported by the :class:`BasePassManager` by
broadcasting the pass runner via the :mod:`qiskit.tools.parallel_map` function.

The base class :class:`BasePassRunner` doesn't define any associated type by itself,
and a developer needs to implement a subclass for a particular object type to optimize.
This `veil of ignorance` allows us to choose the most efficient data representation
for a particular optimization task, while we can reuse the pass flow control machinery
for different input and output types.


Base classes
------------

.. autosummary::
   :toctree: ../stubs/

   BasePassRunner
   BasePassManager

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

Exceptions
----------

.. autoexception:: PassManagerError

"""

from .passrunner import BasePassRunner
from .passmanager import BasePassManager
from .flow_controllers import FlowController, ConditionalController, DoWhileController
from .base_pass import GenericPass
from .propertyset import PropertySet
from .exceptions import PassManagerError
