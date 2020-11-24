# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
================================================
Providers Interface (:mod:`qiskit.providers`)
================================================

.. currentmodule:: qiskit.providers

This module contains the classes used to build external providers for Terra. A
provider is anything that provides an external service to Terra. The typical
example of this is a Backend provider which provides
:class:`~qiskit.providers.Backend` objects which can be used for executing
:class:`~qiskit.circuits.QuantumCircuit` and/or :class:`~qiskit.pulse.Schedule`
objects. This module contains the abstract classes which are used to define the
interface between a provider and terra.

Version Support
===============

Each providers interface abstract class is individually versioned. When we
need to make a change to an interface a new abstract class will be created to
define the new interface. These interface changes are not guaranteed to be
backwards compatible between versions.

Version Changes
----------------

Each minor version release of qiskit-terra **may** increment the version of any
providers interface a single version number. It will be an aggreagate of all
the interface changes for that release on that interface.

Version Support Policy
----------------------

To enable providers to have time to adjust to changes in this interface
Terra will support support multiple versions of each class at once. Given the
nature of one version per release the version deprecation policy is a bit
more conservative than the standard deprecation policy. Terra will support a
provider interface version for a minimum of 3 minor releases or the first
release after 6 months from the release that introduced a version, whichever is
longer, prior to a potential deprecation. After that the standard deprecation
policy will apply to that interface version. This will give providers and users
sufficient time to adapt to potential breaking changes in the interface. So for
example lets say in 0.19.0 ``BackendV2`` is introduced and in the 3 months after
the release of 0.19.0 we release 0.20.0, 0.21.0, and 0.22.0, then 7 months after
0.19.0 we release 0.23.0. In 0.23.0 we can deprecate BackendV2, and it needs to
still be supported and can't be removed until the deprecation policy completes.

It's worth pointing out that Terra's version support policy doesn't mean
providers themselves will have the same support story, they can (and arguably
should) update to newer versions as soon as they can, the support window is
just for Terra's supported versions. Part of this lengthy window prior to
deprecation is to give providers enough time to do their own deprecation of a
potential end user impacting change in a user facing part of the interface
prior to bumping their version. For example, lets say we changed the signature
to `Backend.run()` in ``BackendV34`` in a backwards incompatible way, before
Aer could update its ``AerBackend`` class to use version 34 they'd need to
deprecate the old signature prior to switching over. The changeover for Aer is
not guaranteed to be lockstep with Terra so we need to ensure there is a
sufficient amount of time for Aer to complete it's deprecation cycle prior to
removing version 33 (ie making version 34 mandatory/the minimum version).


Abstract Classes
================

Provider
--------

.. autosummary::
   :toctree: ../stubs/

   Provider
   ProviderV1

Backend
-------

.. autosummary::
   :toctree: ../stubs/

   Backend
   BackendV1

Options
-------

.. autosummary::
   :toctree: ../stubs/

   Options

Job
---

.. autosummary::
   :toctree: ../stubs/

   Job
   JobV1


================================================================
Legacy Provider Interface Base Objects (:mod:`qiskit.providers`)
================================================================

.. currentmodule:: qiskit.providers

Base Objects
============

.. autosummary::
   :toctree: ../stubs/

   BaseProvider
   BaseBackend
   BaseJob

Job Status
==========

.. autosummary::
   :toctree: ../stubs/

   JobStatus

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   QiskitBackendNotFoundError
   BackendPropertyError
   JobError
   JobTimeoutError
"""

import pkgutil

# Providers interface
from qiskit.providers.provider import Provider
from qiskit.providers.provider import ProviderV1
from qiskit.providers.backend import Backend
from qiskit.providers.backend import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.job import Job
from qiskit.providers.job import JobV1
# Legacy providers interface
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.baseprovider import BaseProvider
from qiskit.providers.basejob import BaseJob
from qiskit.providers.exceptions import (JobError, JobTimeoutError, QiskitBackendNotFoundError,
                                         BackendPropertyError, BackendConfigurationError)
from qiskit.providers.jobstatus import JobStatus


# Allow extending this namespace.
__path__ = pkgutil.extend_path(__path__, __name__)
