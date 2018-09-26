# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""

import logging
import warnings
from copy import deepcopy
import uuid

import qiskit.backends.ibmq as ibmq
import qiskit.backends.local as local

from qiskit._qiskiterror import QISKitError
from qiskit._quantumcircuit import QuantumCircuit

from qiskit.transpiler._passmanager import PassManager
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler._transpiler import (_matches_coupling_map,
                                           _pick_best_layout,
                                           _dags_2_qobj_parallel,
                                           _transpile_dags_parallel)
from qiskit.qobj._qobj import Qobj, QobjConfig, QobjHeader
from qiskit.transpiler._transpilererror import TranspilerError
from qiskit.transpiler._parallel import parallel_map
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
        `qiskit.backends.ibmq.IBMQ` instead (`use_account()`) for using IBMQ
        accounts. For custom `Provider`s, please instantiate them directly.
    """
    if provider_class:
        warnings.warn(
            'The global registry of providers and register() is deprecated '
            'since 0.6. Please instantiate "{}()" directly.'.format(provider_class),
            DeprecationWarning)
        return provider_class(*args, **kwargs)
    else:
        warnings.warn('register() will be deprecated after 0.6. Please use the '
                      'qiskit.IBMQ.use_account() method instead.',
                      DeprecationWarning)

    try:
        provider = ibmq.IBMQ.use_account(*args, **kwargs)
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
        `qiskit.backends.ibmq.IBMQ` instead (`remove_account()`).
    """
    warnings.warn('unregister() will be deprecated after 0.6. Please use the '
                  'qiskit.IBMQ.remove_account() method instead.',
                  DeprecationWarning)
    # TODO: provider is no longer a valid argument - signature, noop?


def registered_providers():
    """Return the currently registered providers.

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.backends.ibmq.IBMQ` instead (`list_accounts()`).
    """
    warnings.warn('registered_providers() will be deprecated after 0.6. Please '
                  'use the qiskit.IBMQ.list_accounts() method instead.',
                  DeprecationWarning)
    return ibmq.IBMQ.list_accounts()


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
        `qiskit.backends.ibmq.IBMQ` and `qiskit.backends.local.Aer` instead
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

    backends = local.Aer.backends(**kwargs) + ibmq.IBMQ.backends(**kwargs)
    return [backend.name() for backend in backends]


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
    """
    backends = [get_backend(name) for name in names]
    return _least_busy_instances(backends).name()


def _least_busy_instances(backends):
    """
    Return the least busy available backend for those that
    have a `pending_jobs` in their `status`. Backends such as
    local backends that do not have this are not considered.

    Args:
        names (list[BaseBackend]): backends to choose from

    Returns:
        BaseBackend: the the least busy backend

    Raises:
        QISKitError: if passing a list of backend names that is
            either empty or none have attribute ``pending_jobs``
    """
    try:
        return min([b for b in backends
                    if b.status()['operational'] and 'pending_jobs' in b.status()],
                   key=lambda b: b.status()['pending_jobs'])
    except (ValueError, TypeError):
        raise QISKitError("Can only find least_busy backend from a non-empty list.")


def get_backend(name):
    """
    Return an instance of a `Backend` object from its name identifier.

    Args:
        name(str): unique name of the backend.
    Returns:
        BaseBackend: a Backend instance.

    .. deprecated:: 0.6+
        After 0.6, this function is deprecated. Please use the methods in
        `qiskit.backends.ibmq.IBMQ` and `qiskit.backends.local.Aer` instead
        (`backends()`).
    """
    warnings.warn('get_backend() will be deprecated after 0.6. Please '
                  'use the qiskit.IBMQ.backends() and qiskit.Aer.backends() '
                  'method instead with the "filters" parameter.',
                  DeprecationWarning)
    try:
        return local.Aer.backend(name)
    except KeyError:
        return ibmq.IBMQ.backend(name)


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
        skip_transpiler (bool): If True, bypass most of the compilation process and
            creates a qobj with minimal check nor translation
    Returns:
        Qobj: the qobj to be run on the backends

    Raises:
        TranspilerError: in case of bad compile options, e.g. the hpc options.
    """
    # pylint: disable=redefined-builtin

    # Check for valid parameters for the experiments.
    if hpc is not None and \
            not all(key in hpc for key in ('multi_shot_optimization', 'omp_num_threads')):
        raise TranspilerError('Unknown HPC parameter format!')

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    if isinstance(backend, str):
        try:
            backend = local.Aer.backend(backend)
        except KeyError:
            backend = ibmq.IBMQ.backend(backend)

    pass_manager = None  # default pass manager which executes predetermined passes
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = PassManager()

    backend_conf = backend.configuration()
    backend_name = backend_conf['name']
    basis_gates = basis_gates or backend_conf['basis_gates']
    coupling_map = coupling_map or backend_conf['coupling_map']

    qobj_config = deepcopy(config or {})
    qobj_config.update({'shots': shots,
                        'max_credits': max_credits,
                        'memory_slots': 0})

    qobj = Qobj(qobj_id=qobj_id or str(uuid.uuid4()),
                config=QobjConfig(**qobj_config),
                experiments=[],
                header=QobjHeader(backend_name=backend_name))

    if seed:
        qobj.config.seed = seed

    qobj.experiments = parallel_map(_build_exp_parallel, list(range(len(circuits))),
                                    task_args=(circuits, backend),
                                    task_kwargs={'initial_layout': initial_layout,
                                                 'basis_gates': basis_gates,
                                                 'config': config,
                                                 'coupling_map': coupling_map,
                                                 'seed': seed,
                                                 'pass_manager': pass_manager})

    qobj.config.memory_slots = max(experiment.config.memory_slots for
                                   experiment in qobj.experiments)

    qobj.config.n_qubits = max(experiment.config.n_qubits for
                               experiment in qobj.experiments)

    return qobj


def _build_exp_parallel(idx, circuits, backend, initial_layout=None,
                        basis_gates='u1,u2,u3,cx,id', config=None,
                        coupling_map=None, seed=None, pass_manager=None):
    """Builds a single Qobj experiment.  Usually called in parallel mode.

    Args:
        idx (int): Index of circuit in circuits list.
        circuits (list): List of circuits passed.
        backend (BaseBackend or str): a backend to compile for
        initial_layout (list): initial layout of qubits in mapping
        basis_gates (str): comma-separated basis gate set to compile to
        config (dict): dictionary of parameters (e.g. noise) used by runner
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        seed (int): random seed for simulators
        pass_manager (PassManager): pass manager instance for the tranpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        experiment: An instance of an experiment to be added to a Qobj.
    """

    circuit = circuits[idx]
    dag = DAGCircuit.fromQuantumCircuit(circuit)

    if (initial_layout is None and not backend.configuration()['simulator']
            and not _matches_coupling_map(dag, coupling_map)):
        _initial_layout = _pick_best_layout(dag, backend)
    else:
        _initial_layout = initial_layout

    dag = _transpile_dags_parallel(0, [dag], [_initial_layout],
                                   basis_gates=basis_gates, coupling_map=coupling_map,
                                   seed=seed, pass_manager=pass_manager)

    experiment = _dags_2_qobj_parallel(
        dag, basis_gates=basis_gates, config=config, coupling_map=coupling_map)

    return experiment


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
    """
    # pylint: disable=missing-param-doc, missing-type-doc
    if isinstance(backend, str):
        try:
            backend = local.Aer.backend(backend)
        except KeyError:
            backend = ibmq.IBMQ.backend(backend)

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
