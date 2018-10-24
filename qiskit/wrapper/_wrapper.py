# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""

import logging
import warnings

from qiskit import IBMQ
from qiskit import Aer

from qiskit.backends import ibmq
from qiskit._qiskiterror import QISKitError
from qiskit import transpiler
from qiskit.transpiler._passmanager import PassManager
from ._circuittoolkit import circuit_from_qasm_file, circuit_from_qasm_string


logger = logging.getLogger(__name__)


def register(*args, provider_class=None, **kwargs):
    """
    Authenticate against an online backend provider.
    This is a factory method that returns the provider that gets registered.

    Args:
        args (tuple): positional arguments passed to provider class initialization
        provider_class (BaseProvider): provider class
        kwargs (dict): keyword arguments passed to provider class initialization.
            For the IBMQSingleProvider default this can include things such as:

                * token (str): The token used to register on the online backend such
                    as the quantum experience.
                * url (str): The url used for online backend such as the quantum
                    experience.
                * hub (str): The hub used for online backend.
                * group (str): The group used for online backend.
                * project (str): The project used for online backend.
                * proxies (dict): Proxy configuration for the API, as a dict with
                    'urls' and credential keys.
                * verify (bool): If False, ignores SSL certificates errors.

    Returns:
        BaseProvider: the provider instance that was just registered.

    Raises:
        QISKitError: if the provider could not be registered

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` instead (`enable_account()`) for using IBMQ
        accounts. For custom `Providers`, please instantiate them directly.
    """
    if provider_class:
        warnings.warn(
            'The global registry of providers and register() is deprecated '
            'since 0.6. Please instantiate "{}()" directly.'.format(provider_class),
            DeprecationWarning)
        return provider_class(*args, **kwargs)
    else:
        warnings.warn('register() will be deprecated after 0.6. Please use the '
                      'qiskit.IBMQ.enable_account() method instead.',
                      DeprecationWarning)

    try:
        provider = IBMQ.enable_account(*args, **kwargs)
    except Exception as ex:
        raise QISKitError("Couldn't instantiate provider! Error: {0}".format(ex))

    return provider


def unregister(provider):
    """
    Removes a provider from list of registered providers.

    Note:
        If backend names from provider1 and provider2 were clashing,
        `unregister(provider1)` removes the clash and makes the backends
        from provider2 available.

    Args:
        provider (BaseProvider): the provider instance to unregister
    Raises:
        QISKitError: if the provider instance is not registered

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` instead (`disable_account()`).
    """
    # pylint: disable=unused-argument
    warnings.warn('unregister() will be deprecated after 0.6. Please use the '
                  'qiskit.IBMQ.disable_account() method instead.',
                  DeprecationWarning)


def registered_providers():
    """Return the currently registered providers.

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` instead (`active_accounts()`).
    """
    warnings.warn('registered_providers() will be deprecated after 0.6. Please '
                  'use the qiskit.IBMQ.active_accounts() method instead.',
                  DeprecationWarning)
    return IBMQ.active_accounts()


# Functions for inspecting and retrieving backends.
def available_backends(filters=None, compact=True):
    """
    Return names of backends that are available in the SDK, optionally filtering
    them based on their capabilities.

    Note:
        In order for this function to return online backends, a connection with
        an online backend provider needs to be established by calling the
        `register()` function.

    Note:
        If two or more providers have backends with the same name, those names
        will be shown only once. To disambiguate and choose a backend from a
        specific provider, get the backend from that specific provider.

        Example::

            p1 = register(token1)
            p2 = register(token2)
            execute(circuit, p1.get_backend('ibmq_5_tenerife'))
            execute(circuit, p2.get_backend('ibmq_5_tenerife'))

    Args:
        filters (dict or callable): filtering conditions.
        compact (bool): group backend names based on compact group names.

    Returns:
        list[str]: the names of the available backends.

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` and `qiskit.backends.local.Aer` instead
        (`backends()`).
    """
    warnings.warn('available_backends() will be deprecated after 0.6. Please '
                  'use the qiskit.IBMQ.backends() and qiskit.Aer.backends() '
                  'method instead.',
                  DeprecationWarning)

    if isinstance(filters, dict):
        kwargs = filters
    else:
        kwargs = {'filters': filters}

    ibmq_names = [backend.name() for backend in IBMQ.backends(**kwargs)]
    aer_names = [backend.name() for backend in Aer.backends(**kwargs)]

    if compact:
        # Hack for backwards compatibility: reverse the groups for local.
        aer_groups = Aer.grouped_backend_names()
        reversed_aer_groups = {}
        for group, items in aer_groups.items():
            for alias in items:
                reversed_aer_groups[alias] = group

        aer_names = list(set(reversed_aer_groups[name] for name in aer_names))

    return ibmq_names + aer_names


def least_busy(names):
    """
    Return the least busy available backend for those that
    have a `pending_jobs` in their `status`. Backends such as
    local backends that do not have this are not considered.

    Args:
        names (list[str]): backend names to choose from
                    (e.g. output of ``available_backends()``)

    Returns:
        str: the name of the least busy backend

    Raises:
        QISKitError: if passing a list of backend names that is
            either empty or none have attribute ``pending_jobs``
    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` instead
        (`backends()`).
    """
    backends = [get_backend(name) for name in names]
    warnings.warn('the global least_busy() will be deprecated after 0.6. Please '
                  'use least_busy() imported from qiskit.backends.ibmq',
                  DeprecationWarning)
    return ibmq.least_busy(backends).name()


def get_backend(name):
    """
    Return an instance of a `Backend` object from its name identifier.

    Args:
        name(str): unique name of the backend.
    Returns:
        BaseBackend: a Backend instance.

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.IBMQ` and `qiskit.Aer` instead
        (`backends()`).
    """
    warnings.warn('the global get_backend() will be deprecated after 0.6. Please '
                  'use the qiskit.IBMQ.backends() and qiskit.Aer.backends() '
                  'method instead with the "name" parameter.'
                  '(or qiskit.IBMQ.get_backend() and qiskit.Aer.get_backend())',
                  DeprecationWarning)
    try:
        return Aer.get_backend(name)
    except KeyError:
        return IBMQ.get_backend(name)


# Functions for compiling and executing.


def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_transpiler=False):
    """Compile a list of circuits into a qobj.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend or str): a backend to compile for
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        qobj_id (int): identifier for the generated qobj
        hpc (dict): HPC simulator parameters
        skip_transpiler (bool): skip most of the compile steps and produce qobj directly

    Returns:
        Qobj: the qobj to be run on the backends

    Raises:
        TranspilerError: in case of bad compile options, e.g. the hpc options.

    .. deprecated:: 0.6+
        After 0.6, compile will only take a backend object.
    """
    # pylint: disable=redefined-builtin
    if isinstance(backend, str):
        warnings.warn('compile() no longer takes backend string names.'
                      'Please pass backend objects, obtained via'
                      'IBMQ.get_backend() or Aer.get_backend().', DeprecationWarning)
        try:
            backend = Aer.get_backend(backend)
        except KeyError:
            backend = IBMQ.get_backend(backend)

    pass_manager = None  # default pass manager which executes predetermined passes
    # TODO (jay) why do we need to pass skip and not pass manager directly
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = PassManager()

    qobj_standard = transpiler.compile(circuits, backend, config, basis_gates, coupling_map,
                                       initial_layout, shots, max_credits, seed, qobj_id, hpc,
                                       pass_manager)
    return qobj_standard


def execute(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_transpiler=False):
    """Executes a set of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend or str): a backend to execute the circuits on
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        qobj_id (int): identifier for the generated qobj
        hpc (dict): HPC simulator parameters
        skip_transpiler (bool): skip most of the compile steps and produce qobj directly

    Returns:
        BaseJob: returns job instance derived from BaseJob

    .. deprecated:: 0.6+
        After 0.6, execute will only take a backend object, not string.
    """
    # pylint: disable=missing-param-doc, missing-type-doc
    if isinstance(backend, str):
        warnings.warn('execute() no longer takes backend string names. '
                      'Please pass backend objects, obtained via'
                      'IBMQ.get_backend() or Aer.get_backend().', DeprecationWarning)
        try:
            backend = Aer.get_backend(backend)
        except KeyError:
            backend = IBMQ.get_backend(backend)

    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id, hpc,
                   skip_transpiler)
    return backend.run(qobj)


# Functions for importing qasm


def load_qasm_string(qasm_string, name=None,
                     basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                                 "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap"):
    """Construct a quantum circuit from a qasm representation (string).

    Args:
        qasm_string (str): a string of qasm, or a filename containing qasm.
        basis_gates (str): basis gates for the quantum circuit.
        name (str or None): the name of the quantum circuit after loading qasm
            text into it. If no name given, assign automatically.
    Returns:
        QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the string is not valid QASM
    """
    return circuit_from_qasm_string(qasm_string, name, basis_gates)


def load_qasm_file(qasm_file, name=None,
                   basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                               "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap"):
    """Construct a quantum circuit from a qasm representation (file).

    Args:
        qasm_file (str): a string for the filename including its location.
        name (str or None): the name of the quantum circuit after
            loading qasm text into it. If no name is give the name is of
            the text file.
        basis_gates (str): basis gates for the quantum circuit.
    Returns:
         QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the file cannot be read.
    """
    return circuit_from_qasm_file(qasm_file, name, basis_gates)


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    if qobj.experiments:
        circuits = []
        for x in qobj.experiments:
            if hasattr(x.header, 'compiled_circuit_qasm'):
                circuits.append(
                    load_qasm_string(x.header.compiled_circuit_qasm))
        return circuits
    # TODO(mtreinish): add support for converting a qobj if the qasm isn't
    # embedded in the header
    return None
