from concurrent import futures
from threading import Lock
import sys
import time

import qiskit.backends
from qiskit._result import Result
from qiskit._resulterror import ResultError
from qiskit import QISKitError
from qiskit import _openquantumcompiler as openquantumcompiler
from IBMQuantumExperience.IBMQuantumExperience import (IBMQuantumExperience)

def run_local_backend(qobj):
    """Run a program of compiled quantum circuits on the local machine.

    Args:
        qobj (dict): quantum object dictionary

    Returns:
        Result object.
    """
    for circuit in qobj['circuits']:
        if circuit['compiled_circuit'] is None:
            compiled_circuit = openquantumcompiler.compile(circuit['circuit'],
                                                           format='json')
            circuit['compiled_circuit'] = compiled_circuit
    BackendClass = qiskit.backends.get_backend_class(qobj['config']['backend'])
    backend = BackendClass(qobj)
    return backend.run()

def run_remote_backend(qobj, wait=5, timeout=60, silent=True):
    """
    Args:
        qobj (dict): quantum object dictionary
        wait (float): seconds between run attempts
        timeout (float): seconds
    Returns:
        Result object.
    Raises:
        QISKitError: if "ERROR" string in server response.
    """
    BackendClass = qiskit.backends.get_backend_class(qobj['config']['backend'])
    backend = BackendClass(qobj)
    return backend.run(wait=wait, timeout=timeout, silent=silent)

def remote_backends(api):
    """Get the remote backends.

    Queries network API if it exists and gets the backends that are online.

    Returns:
        List of online backends if the online api has been set or an empty
        list of it has not been set.
    """
    return [backend['name'] for backend in api.available_backends()]

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
                results: A list of Result objects.
            max_workers (int): The maximum number of workers to use.
            token (str): Server API token
            url (str): Server URL.
            api (IBMQuantumExperience): API instance to use. If set,
                /token/ and /url/ are ignored.
        """
        self.q_jobs = q_jobs
        self.max_workers = max_workers
        # check whether any jobs are remote
        self._local_backends = qiskit.backends.local_backends()
        self.online = any(qj.backend not in self._local_backends for qj in q_jobs)
        self.futures = {}
        self.lock = Lock()
        # Set a default dummy callback just in case the user doesn't want
        # to pass any callback.
        self.callback = (lambda rs: ()) if callback is None else callback
        self.num_jobs = len(self.q_jobs)
        self.jobs_results = []
        if self.online:
            self._api = api if api else IBMQuantumExperience(token,
                                                             {"url": url},
                                                             verify=True)
            self._online_backends = remote_backends(self._api)
            # Check for the existance of the backend
            for q_job in q_jobs:
                if q_job.backend not in self._online_backends + self._local_backends:
                    raise QISKitError("Backend %s not found!" % q_job.backend)

            self._api_config = {}
            self._api_config["token"] = token
            self._api_config["url"] = {"url": url}
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
                             'result': ex},
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

    def submit(self, wait=5, timeout=120, silent=True):
        """Process/submit jobs

        Args:
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time waiting for the results
            silent (bool): If true, prints out results
        """
        executor = self.executor_class(max_workers=self.max_workers)
        for q_job in self.q_jobs:
            if q_job.backend in self._local_backends:
                future = executor.submit(run_local_backend,
                                         q_job.qobj)
            elif self.online and q_job.backend in self._online_backends:
                future = executor.submit(run_remote_backend,
                                         q_job.qobj,
                                         wait=wait, timeout=timeout,
                                         silent=silent)
            future.silent = silent
            future.qobj = q_job.qobj
            self.futures[future] = q_job.qobj
            future.add_done_callback(self._job_done_callback)
