from concurrent import futures
import logging
import pprint
from threading import Lock
import sys
import time

import qiskit.backends
from qiskit.backends import (local_backends, remote_backends)
from qiskit._result import Result
from qiskit._resulterror import ResultError
from qiskit import QISKitError
from qiskit import _openquantumcompiler as openquantumcompiler
from IBMQuantumExperience.IBMQuantumExperience import (IBMQuantumExperience)

logger = logging.getLogger(__name__)

def run_backend(q_job):
    """Run a program of compiled quantum circuits on the local machine.

    Args:
        q_job (QuantumJob): job object

    Returns:
        Result object.
    """
    backend_name = q_job.backend
    qobj = q_job.qobj
    if backend_name in local_backends(): # remove condition when api gets qobj
        for circuit in qobj['circuits']:
            if circuit['compiled_circuit'] is None:
                compiled_circuit = openquantumcompiler.compile(circuit['circuit'],
                                                               format='json')
                circuit['compiled_circuit'] = compiled_circuit
    backend = qiskit.backends.get_backend_instance(backend_name)
    return backend.run(q_job)

class JobProcessor():
    """
    process a bunch of jobs and collect the results
    """

    def __init__(self, q_jobs, callback, max_workers=1):
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
        self.online = any(qj.backend not in local_backends() for qj in q_jobs)
        self.futures = {}
        self.lock = Lock()
        # Set a default dummy callback just in case the user doesn't want
        # to pass any callback.
        self.callback = (lambda rs: ()) if callback is None else callback
        self.num_jobs = len(self.q_jobs)
        self.jobs_results = []
        if self.online:
            # verify backends across all jobs
            for q_job in q_jobs:
                if q_job.backend not in remote_backends() + local_backends():
                    raise QISKitError("Backend %s not found!" % q_job.backend)
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
            logger.info(pprint.pformat(result))
            self.callback(self.jobs_results)

    def submit(self):
        """Process/submit jobs"""
        executor = self.executor_class(max_workers=self.max_workers)
        for q_job in self.q_jobs:
            future = executor.submit(run_backend, q_job)
            future.qobj = q_job.qobj
            self.futures[future] = q_job.qobj
            future.add_done_callback(self._job_done_callback)
