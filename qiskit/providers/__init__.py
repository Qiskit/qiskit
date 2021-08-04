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
=============================================
Providers Interface (:mod:`qiskit.providers`)
=============================================

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
providers interface a single version number. It will be an aggregate of all
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

======================
Writing a New Provider
======================

If you have a quantum device or simulator that you would like to integrate with
Qiskit you will need to write a provider. A provider will provide Terra with a
method to get available :class:`~qiskit.provider.BackendV1` objects. The
:class:`~qiskit.provider.BackendV1` object provides both information describing
a backend and its operation for the :mod:`~qiskit.transpiler` so that circuits
can be compiled to something that is optimized and can execute on the
backend. It also provides the :meth:`~qiskit.providers.BackendV1.run` method which can
run the :class:`~qiskit.circuit.QuantumCircuit` objects and/or
:class:`~qiskit.pulse.Schedule` objects. This enables users and other Qiskit
APIs, such as :func:`~qiskit.execute.execute` and higher level algorithms in
Qiskit Aqua, to get results from executing circuits on devices in a standard
fashion regardless of how the Backend is implemented. At a high level the basic
steps for writing a provider are:

 * Implement a :class:`~qiskit.providers.ProviderV1` subclass that handles
   access to the backend(s).
 * Implement a :class:`~qiskit.providers.BackendV1` subclass and its
   :meth:`~~qiskit.providers.BackendV1.run` method.

   * Add any custom gates for the backend's basis to the session
     :class:`~qiskit.circuit.EquivalenceLibrary` instance.

 * Implement a :class:`~qiskit.providers.JobV1` subclass that handles
   interacting with a running job.

For a simple example of a provider the
`qiskit-aqt-provider <https://github.com/qiskit-community/qiskit-aqt-provider>`__
package is worth using as an example.

Provider
========

A provider class serves a single purpose: to get backend objects that enable
executing circuits on a device or simulator. The expectation is that any
required credentials and/or authentication will be handled in the initialization
of a provider object. The provider object will then provide a list of backends,
and methods to filter and acquire backends (using the provided credentials if
required). An example provider class looks like::

    from qiskit.providers import ProviderV1 as Provider
    from qiskit.providers.providerutils import filter_backends

    from .backend import MyBackend

    class MyProvider(Provider):

        def __init__(self, token=None):
            super().__init__()
            self.token = token
            self.backends = [MyBackend(provider=self)]

        def backends(self, name=None, **kwargs):
            if name:
                backends = [
                    backend for backend in backends if backend.name() == name]
            return filter_backends(backends, filters=filters, **kwargs)

Ensure that any necessary information for
authentication (if required) are present in the class and that the backends
method matches the required interface. The rest is up to the specific provider on how to implement.

Backend
=======

The backend classes are the core to the provider. These classes are what
provide the interface between Qiskit and the hardware or simulator that will
execute circuits. This includes providing the necessary information to
describe a backend to the compiler so that it can embed and optimize any
circuit for the backend. There are 3 required things in every backend object: a
backend configuration property, a ``run()`` method, and a
``_default_options()`` method. For example, a minimum working example would be
something like::

    from qiskit.providers import BackendV1 as Backend
    from qiskit.providers.models import BackendConfiguration
    from qiskit.providers import Options


    class Mybackend(Backend):

        configuration = {
            'backend_name': 'my_backend',
            'backend_version': '1.0.0',
            'n_qubits': 5,
            'basis_gates': ['p', 'sx', 'u', 'cx', 'id'],
            'coupling_map': [[0, 1], [1, 2], [2, 3], [3, 4]],
            'simulator': False,
            'local': False,
            'conditional': True,
            'open_pulse': False,
            'memory': True,
            'max_shots': 8196,
            'gates': []
        }

        def __init(self, provider):
            super().__init__(
                configuration=BackendConfiguration.from_dict(self.configuration),
                provider=provider)

        @classmethod
        def _default_options(cls):
            return Options(shots=1024, memory=False)

        def run(circuits, **kwargs):
            # serialize circuits submit to backend and create a job
            for kwarg in kwargs:
                if not hasattr(kwarg, self.options):
                    warnings.warn(
                        "Option %s is not used by this backend" % kwarg,
                        UserWarning, stacklevel=2)
            options = {
                'shots': kwargs.get('shots', self.options.shots),
                'memory': kwargs.get('memory', self.options.shots),
            }
            job_json = convert_to_wire_format(circuit, options)
            job_handle = submit_to_backend(job_jsonb)
            return MyJob(self. job_handle, job_json, circuit)


Transpiler Interface
--------------------

The key piece of the backend object is how it describes itself to the compiler.
This is composed of three classes, the
:class:`~qiskit.providers.models.BackendConfiguration` which describes the
immutable properties of the backend (things like the number of qubits, coupling
map, etc), the :class:`~qiskit.providers.models.BackendProperties` which
describes measured properties of the device (such as current error rates), and
the :class:`~qiskit.providers.models.PulseDefaults` which defines the
default pulse behavior for the backend (such as the pulse sequences for gates).
Of these only :class:`~qiskit.providers.models.BackendConfiguration` is
required.

It's worth noting that this interface will likely evolve over time as the
compiler grows and improves, thus requiring new information from the backends.
This is a key reason why the providers interfaces are versioned to enable
making these changes in a controlled manner. The details here will change and
reflect the latest version of this interface.

Custom Basis Gates
^^^^^^^^^^^^^^^^^^

1. If your backend doesn't use gates in the Qiskit circuit library
(:mod:`qiskit.circuit.library`) you can integrate support for this into your
provider. The basic method for doing this is first to define a
:class:`~qiskit.circuit.Gate` class for all the custom gates in the basis
set. For example::

    import numpy as np

    from qiskit.circuit import Gate
    from qiskit.circuit import QuantumCircuit

    class SYGate(Gate):
        def __init__(self, label=None):
            super().__init__("sy", 1, [], label=label)

        def _define(self):
            qc = QuantumCircuit(1)
            q.ry(np.pi / 2, 0)
            self.definition = qc

The key thing to ensure is that for any custom gates in your Backend's basis set
your custom gate's name attribute (the first param on
``super().__init__()`` in the ``__init__`` definition above) does not conflict
with the name of any other gates. The name attribute is what is used to
identify the gate in the basis set for the transpiler. If there is a conflict
the transpiler will not know which gate to use.

2. After you've defined the custom gates to use for the Backend's basis set
then you need to add equivalence rules to the standard equivalence library
so that the :func:`~qiskit.compiler.transpile` function and
:mod:`~qiskit.transpiler` module can convert an arbitrary circuit using the
custom basis set. This can be done by defining equivalent circuits, in terms
of the custom gate, for standard gates. Typically if you can convert from a
:class:`~qiskit.circuit.library.CXGate` (if your basis doesn't include a
standard 2 qubit gate) and some commonly used single qubit rotation gates like
the :class:`~qiskit.ciruit.library.HGate` and
:class:`~qiskit.circuit.library.UGate` that should be sufficient for the
transpiler to translate any circuit into the custom basis gates. But, the more
equivalence rules that are defined from standard gates to your basis the more
efficient translation from an arbitrary circuit to the target basis will be
(although not always, and there is a diminishing margin of return).

For example, if you were to add some rules for the above custom ``SYGate``
we could define the :class:`~qiskit.circuit.library.U2Gate` and
:class:`~qiskit.circuit.library.HGate`::

    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.circuit.library import HGate
    from qiskit.circuit.library import ZGate
    from qiskit.circuit.library import RZGate
    from qiskit.circuit.library import U2Gate


    # H => Z SY
    q = qiskit.QuantumRegister(1, "q")
    def_sy_h = qiskit.QuantumCircuit(q)
    def_sy_h.append(ZGate(), [q[0]], [])
    def_sy_h.append(SYGate(), [q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(
        HGate(), def_sy_h)

    # u2 => Z SY Z
    phi = qiskit.circuit.Parameter('phi')
    lam = qiskit.circuit.Parameter('lambda')
    q = qiskit.QuantumRegister(1, "q")
    def_sy_u2 = qiskit.QuantumCircuit(q)
    def_sy_u2.append(RZGate(lam), [q[0]], [])
    def_sy_u2.append(SYGate(), [q[0]], [])
    def_sy_u2.append(RZGate(phi), [q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(
        U2Gate(phi, lam), def_sy_u2)


You will want this to be run on import so that as soon as the provider's
package is imported it will be run. This will ensure that any time the
:class:`~qiskit.transpiler.passes.BasisTranslator` pass is run with the
custom gates the equivalence rules are defined.

It's also worth noting that depending on the basis you're using, some optimization
passes in the transpiler, such as
:class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition`, may not
be able to operate with your custom basis. For our ``SYGate`` example, the
:class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition` will not be
able to simplify runs of single qubit gates into the SY basis. This is because
the :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` class does not
know how to work in the SY basis. To solve this the ``SYGate`` class would
need to be added to Qiskit and
:class:`~qiskit.quantum_info.OneQubitEulerDecomposer` updated to support
decomposing to the ``SYGate``. Longer term that is likely a better direction
for custom basis gates and contributing the definitions and support in the
transpiler will ensure that it continues to be well supported by Qiskit
moving forward.

Run Method
----------

Of key importance is the :meth:`~qiskit.providers.BackendV1.run` method, which
is used to actually submit circuits to a device or simulator. The run method
handles submitting the circuits to the backend to be executed and returning a
:class:`~qiskit.providers.Job` object. Depending on the type of backend this
typically involves serializing the circuit object into the API format used by a
backend. For example, on IBMQ backends from the ``qiskit-ibmq-provider``
package this involves converting from a quantum circuit and options into a
`qobj <https://arxiv.org/abs/1809.03452>`__ JSON payload and submitting
that to the IBM Quantum API. Since every backend interface is different (and
in the case of the local simulators serialization may not be needed) it is
expected that the backend's ``run()`` method will handle this conversion.

An example run method would be something like::

    def run(self, circuits. **kwargs):
        for kwarg in kwargs:
            if not hasattr(kwarg, self.options):
                warnings.warn(
                    "Option %s is not used by this backend" % kwarg,
                    UserWarning, stacklevel=2)
        options = {
            'shots': kwargs.get('shots', self.options.shots)
            'memory': kwargs.get('memory', self.options.shots),
        }
        job_json = convert_to_wire_format(circuit, options)
        job_handle = submit_to_backend(job_jsonb)
        return MyJob(self. job_handle, job_json, circuit)

Options
-------

There are often several options for a backend that control how a circuit is run.
The typical example of this is something like the number of ``shots`` which is
how many times the circuit is to be executed. The options available for a
backend are defined using an :class:`~qiskit.providers.Options` object. This
object is initially created by the ``_default_options`` method of a Backend
class. The default options returns an initialized
:class:`~qiskit.providers.Options` object with all the default values for all
the options a backend supports. For example, if the backend supports only
supports ``shots`` the ``_default_options`` method would look like::

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)


Job
===

The output from the ``run()`` method is a :class:`~qiskit.providers.JobV1`
object. Each provider is expected to implement a custom job subclass that
defines the behavior for the provider. There are 2 types of jobs depending on
the backend's execution method, either a sync or async. By default jobs are
considered async and the expectation is that it represents a handle to the
async execution of the circuits submitted with ``Backend.run()``. An async
``Job`` object provides users the ability to query the status of the execution,
cancel a running job, and block until the execution is finished. The
:class:`~qiskit.providers.JobV1.result` is the primary user facing method
which will block until the execution is complete and then will return a
:class:`~qiskit.result.Result` object with results of the job.

For some backends (mainly local simulators) the execution of circuits is a
synchronous operation and there is no need to return a handle to a running job
elsewhere. For sync jobs its expected that the ``run()`` method on the backend
will block until a :class:`~qiskit.result.Result` object is generated and the
sync job will return with that inner :class:`~qiskit.result.Result` object.

An example job class for an async API based backend would look something like::

    from qiskit.providers import JobV1 as Job
    from qiskit.providers import JobError
    from qiskit.providers import JobTimeoutError
    from qiskit.providers.jobstatus import JobStatus
    from qiskit.result import Result


    class MyJob(Job):
        def __init__(self, backend, job_id, job_json, circuits):
            super().__init__(backend, job_id)
            self._backend = backend
            self.job_json = job_json
            self.circuits = circuits

        def _wait_for_result(self, timeout=None, wait=5):
            start_time = time.time()
            result = None
            while True:
                elapsed = time.time() - start_time
                if timeout and elapsed >= timeout:
                    raise JobTimeoutError('Timed out waiting for result')
                result = get_job_status(self._job_id)
                if result['status'] == 'complete':
                    break
                if result['status'] == 'error':
                    raise JobError('Job error')
                time.sleep(wait)
            return result

        def result(self, timeout=None, wait=5):
            result = self._wait_for_result(timeout, wait)
            results = [{'success': True, 'shots': len(result['counts']),
                        'data': result['counts']}]
            return Result.from_dict({
                'results': results,
                'backend_name': self._backend.configuration().backend_name,
                'backend_version': self._backend.configuration().backend_version,
                'job_id': self._job_id,
                'qobj_id': ', '.join(x.name for x in self.circuits),
                'success': True,
            })

        def status(self):
            result = get_job_status(self._job_id)
            if result['status'] == 'running':
                status = JobStatus.RUNNING
            elif result['status'] == 'complete':
                status = JobStatus.DONE
            else:
                status = JobStatus.ERROR
            return status

    def submit(self):
        raise NotImplementedError

and for a sync job::

    class MySyncJob(Job):
        _async = False

        def __init__(self, backend, job_id, result):
            super().__init__(backend, job_id)
            self._result = result

        def submit(self):
            return

        def result(self):
            return self._result

        def status(self):
            return JobStatus.DONE


================================================================
Legacy Provider Interface Base Objects (:mod:`qiskit.providers`)
================================================================

These abstract interfaces are deprecated and will be removed in a future
release. The documentation here is left for reference purposes while they're
still supported, but if you're creating or maintaining a provider you should
be using the versioned interface.

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
from qiskit.providers.exceptions import (
    JobError,
    JobTimeoutError,
    QiskitBackendNotFoundError,
    BackendPropertyError,
    BackendConfigurationError,
)
from qiskit.providers.jobstatus import JobStatus


# Allow extending this namespace.
__path__ = pkgutil.extend_path(__path__, __name__)
