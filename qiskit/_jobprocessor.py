from concurrent import futures
from threading import Lock
import sys
import time
import random
import string
from qiskit._result import Result
import qiskit.backends as backends
from qiskit import QISKitError
from qiskit import _openquantumcompiler as openquantumcompiler
from IBMQuantumExperience.IBMQuantumExperience import (IBMQuantumExperience,
                                                       ApiError)

def run_local_backend(qobj):
    """Run a program of compiled quantum circuits on the local machine.

    Args:
        qobj (dict): quantum object dictionary

    Returns:
      Dictionary of form,
      job_result = {
        "status": DATA,
        "result" : [
          {
            "data": DATA,
            "status": DATA,
          },
            ...
        ]
        "name": DATA,
        "backend": DATA
      }
    """
    for circ in qobj['circuits']:
        if circ['compiled_circuit'] is None:
            compiled_circuit = openquantumcompiler.compile(circ['circuit'])
            circ['compiled_circuit'] = openquantumcompiler.dag2json(compiled_circuit)
    BackendClass = backends.get_backend_class(qobj['config']['backend'])
    backend = BackendClass(qobj)
    return backend.run()

def run_remote_backend(qobj, api, wait=5, timeout=60, silent=True):
    """
    Args:
        qobj (dict): quantum object dictionary
        api (IBMQuantumExperience): IBMQuantumExperience API connection

    Raises:
        QISKitError: if "ERROR" string in server response.
    """
    jobs = []
    for job in qobj['circuits']:
        if (('compiled_circuit_qasm' not in job) or
            (job['compiled_circuit_qasm'] is None)):
            compiled_circuit = openquantumcompiler.compile(job['circuit'].qasm())
            job['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)
        if isinstance(job['compiled_circuit_qasm'], bytes):
            jobs.append({'qasm': job['compiled_circuit_qasm'].decode()})
        else:
            jobs.append({'qasm': job['compiled_circuit_qasm']})

    seed0 = qobj['circuits'][0]['config']['seed']
    output = api.run_job(jobs, qobj['config']['backend'],
                         shots=qobj['config']['shots'],
                         max_credits=qobj['config']['max_credits'],
                         seed=seed0)
    if 'ERROR' in output:
        raise QISKitError(output['ERROR'])
    job_result = _wait_for_job(output['id'], api, wait=wait, timeout=timeout, silent=silent)
    job_result['name'] = qobj['id']
    job_result['backend'] = qobj['config']['backend']
    this_result = Result(job_result, qobj)
    return this_result

def _wait_for_job(jobid, api, wait=5, timeout=60, silent=True):
    """Wait until all online ran circuits of a qobj are 'COMPLETED'.

    Args:
        jobid:  is a list of id strings.
        api (IBMQuantumExperience): IBMQuantumExperience API connection
        wait (int):  is the time to wait between requests, in seconds
        timeout (int):  is how long we wait before failing, in seconds
        silent (bool): is an option to print out the running information or
            not

    Returns:
        A list of results that correspond to the jobids.

    Raises:
        QISKitError:
    """
    timer = 0
    timeout_over = False
    job_result = api.get_job(jobid)
    if 'status' not in job_result:
        from pprint import pformat
        raise QISKitError("get_job didn't return status: %s" % (pformat(job_result)))

    while job_result['status'] == 'RUNNING':
        if timer >= timeout:
            return {'status': 'ERROR', 'result': 'Time Out'}
        time.sleep(wait)
        timer += wait
        if not silent:
            print('status = %s (%d seconds)' % (job_result['status'], timer))
        job_result = api.get_job(jobid)

        if 'status' not in job_result:
            from pprint import pformat
            raise QISKitError("get_job didn't return status: %s" % (pformat(job_result)))
        if job_result['status'] == 'ERROR_CREATING_JOB' or job_result['status'] == 'ERROR_RUNNING_JOB':
            return {'status': 'ERROR', 'result': job_result['status']}

    # Get the results
    job_result_return = []
    for index in range(len(job_result['qasms'])):
        job_result_return.append({'data': job_result['qasms'][index]['data'],
                                  'status': job_result['qasms'][index]['status']})
    return {'status': job_result['status'], 'result': job_result_return}

def remote_backends(api):
    """Get the remote backends.

    Queries network API if it exists and gets the backends that are online.

    Returns:
        List of online backends if the online api has been set or an empty
        list of it has not been set.
    """
    return [backend['name'] for backend in api.available_backends() ]

class QuantumJob():
    """Creates a quantum circuit job"""

    def __init__(self, circuits, backend='local_qasm_simulator',
                 circuit_configs=None, timeout=60, seed=None,
                 resources={'max_credits': 3}, shots=1024, names=None,
                 doCompile=False, preformatted=False):
        """
        Args:
            circuit (QuantumCircuit | qobj): QuantumCircuit or list of QuantumCircuit
                objects for job.
            backend (str): the backend to run the circuit on.
            resources (dict): resource requirements of job.
            timeout (float): timeout for job in seconds.
            coupling_map (dict): A directed graph of coupling::

                {
                 control(int):
                     [
                         target1(int),
                         target2(int),
                         , ...
                    ],
                     ...
                }

                eg. {0: [2], 1: [2], 3: [2]}

            initial_layout (dict): A mapping of qubit to qubit::

                                  {
                                    ("q", strart(int)): ("q", final(int)),
                                    ...
                                  }
                                  eg.
                                  {
                                    ("q", 0): ("q", 0),
                                    ("q", 1): ("q", 1),
                                    ("q", 2): ("q", 2),
                                    ("q", 3): ("q", 3)
                                  }
            shots (int): the number of shots
            max_credits (int): the max credits to use 3, or 5
            seed (int): the intial seed the simulatros use
            circuit_type (str): "compiled_dag" or "uncompiled_dag" or
                "quantum_circuit"
            preformated (bool): the objects in circuits are already compiled
                and formatted (qasm for online, json for local). If true the
                parameters "names" and "circuit_configs" must also be defined
                of the same length as "circuits".
        """
        if isinstance(circuits, list):
            self.circuits = circuits
        else:
            self.circuits = [circuits]
        if names is None:
            self.names = []
            for circuit in range(len(self.circuits)):
                self.names.append(
                    ''.join([random.choice(string.ascii_letters +
                                           string.digits)
                             for i in range(10)]))
        elif isinstance(names, list):
            self.names = names
        else:
            self.names = [names]
        self._local_backends = backends.local_backends()
        self.timeout = timeout
        # check whether circuits have already been compiled
        # and formatted for backend.
        if preformatted:
            self.qobj = circuits
            self.backend = self.qobj['config']['backend']
            self.resources = {'max_credits':
                              self.qobj['config']['max_credits']}
        else:
            self.backend = backend
            self.resources = resources
            # local and remote backends currently need different
            # compilied circuit formats
            formatted_circuits = []
            if doCompile:
                for circuit in self.circuits:
                    formatted_circuits.append(None)
            else:
                if backend in self._local_backends:
                    for circuit in self.circuits:
                        formatted_circuits.append(openquantumcompiler.dag2json(circuit))
                else:
                    for circuit in self.circuits:
                        formatted_circuits.append(circuit.qasm(qeflag=True))
            # create circuit component of qobj
            circuitRecords = []
            if circuit_configs is None:
                config = {'coupling_map': None,
                          'basis_gates': 'u1,u2,u3,cx,id',
                          'layout': None,
                          'seed': seed}
                circuit_configs = [config] * len(self.circuits)
            for circuit, fcircuit, name, config in zip(self.circuits,
                                                       formatted_circuits,
                                                       self.names,
                                                       circuit_configs):
                record = {
                    'name': name,
                    'compiled_circuit': None if doCompile else fcircuit,
                    'compiled_circuit_qasm': None if doCompile else fcircuit,
                    'circuit': circuit,
                    'config': config
                }
                circuitRecords.append(record)
            qobjid = ''.join([random.choice(
                string.ascii_letters + string.digits) for i in range(10)])
            self.qobj = {'id': qobjid,
                         'config': {
                             'max_credits': resources['max_credits'],
                             'shots': shots,
                             'backend': backend
                         },
                         'circuits': circuitRecords
            }
        self.seed = seed
        self.result = None
        self.doCompile = doCompile


class JobProcessor():
    """
    process a bunch of jobs and collect the results
    """
    def __init__(self, q_jobs, callback, max_workers=1, token=None, url=None, api=None):
        """
        Args:
            q_jobs (list(QuantumJob)): List of QuantumJob objects.
            callback (fn(results)): The function that will be called when all
                jobs finish. The signature of the function must be:
                fn(results)
                results: A list of (result, qobj) tuples.
            max_workers (int): The maximum number of workers to use.
            token (str): Server API token
            url (str): Server URL.
            api (IBMQuantumExperience): API instance to use. If set,
                /token/ and /url/ are ignored.
        """
        self.q_jobs = q_jobs
        self.max_workers = max_workers
        # check whether any jobs are remote
        self._local_backends = backends.local_backends()
        self.online = any(qj.backend not in self._local_backends for qj in q_jobs)
        self.futures = {}
        self.lock = Lock()
        # Set a default dummy callback just in case the user doesn't want
        # to pass any callback.
        self.callback = (lambda rs:()) if callback is None else callback
        self.num_jobs = len(self.q_jobs)
        self.jobs_results = []
        if self.online:
            self._api = api if api else IBMQuantumExperience(token,
                                                             {"url": url},
                                                             verify=True)
            self._online_backends = remote_backends(self._api)
            self._api_config = {}
            self._api_config["token"] = token
            self._api_config["url"] =  {"url": url}
        else:
            self._api = None
            self._online_backends = None
            self._api_config = None
        if self.online:
            # I/O intensive -> use ThreadedPoolExecutor
            self.executor_class = futures.ThreadPoolExecutor
        else:
            # CPU intensive -> use ProcessPoolExecutor
            self.executor_class = futures.ProcessPoolExecutor

    def _job_done_callback(self, future):
        try:
            result = future.result()
        except Exception as ex:
            result = Result({'status': 'ERROR',
                             'result': [str(ex)]},
                            future.qobj)
        with self.lock:
            self.futures[future]['result'] = result
            self.jobs_results.append(result)
            if self.num_jobs != 0:
                self.num_jobs -= 1
        # Call the callback when all jobs have finished
        if self.num_jobs == 0:
            if not future.silent:
                import pprint
                pprint.pprint(result)
                sys.stdout.flush()
            self.callback(self.jobs_results)

    def submit(self, silent=True):
        """Process/submit jobs

        Args:
            silent (bool): print results if true.
        """
        executor = self.executor_class(max_workers=self.max_workers)
        for q_job in self.q_jobs:
            if q_job.backend in self._local_backends:
                future = executor.submit(run_local_backend,
                                         q_job.qobj)
            elif self.online and q_job.backend in self._online_backends:
                future = executor.submit(run_remote_backend,
                                         q_job.qobj,
                                         self._api)
            future.silent = silent
            future.qobj = q_job.qobj
            future.add_done_callback(self._job_done_callback)
            self.futures[future] = q_job.qobj
