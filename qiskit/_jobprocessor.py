from concurrent import futures
from threading import Lock
import sys
import time

from IBMQuantumExperience import IBMQuantumExperience

from qiskit._result import Result
from qiskit._resulterror import ResultError
# Stable Modules
from qiskit import QISKitError
# Local Simulator Modules
from qiskit import simulators
# compiler module
from qiskit import _openquantumcompiler as openquantumcompiler

def run_local_simulator(qobj):
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
    for circuit in qobj['circuits']:
        if circuit['compiled_circuit'] is None:
            compiled_circuit = openquantumcompiler.compile(circuit['circuit'],
                                                           format='json')
            circuit['compiled_circuit'] = compiled_circuit
    local_simulator = simulators.LocalSimulator(qobj)
    local_simulator.run()
    return local_simulator.result()

def run_remote_backend(qobj, api, wait=5, timeout=60, silent=True):
    """
    Args:
        qobj (dict): quantum object dictionary
        api (IBMQuantumExperience): IBMQuantumExperience API connection

    Raises:
        QISKitError: if "ERROR" string in server response.
    """
    api_jobs = []
    for circuit in qobj['circuits']:
        if (('compiled_circuit_qasm' not in circuit) or
            (circuit['compiled_circuit_qasm'] is None)):
            compiled_circuit = openquantumcompiler.compile(
                circuit['circuit'].qasm())
            circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)
        if isinstance(circuit['compiled_circuit_qasm'], bytes):
            api_jobs.append({'qasm': circuit['compiled_circuit_qasm'].decode()})
        else:
            api_jobs.append({'qasm': circuit['compiled_circuit_qasm']})

    seed0 = qobj['circuits'][0]['config']['seed']
    output = api.run_job(api_jobs, qobj['config']['backend'],
                         shots=qobj['config']['shots'],
                         max_credits=qobj['config']['max_credits'],
                         seed=seed0)
    if 'error' in output:
        raise ResultError(output['error'])

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

def local_backends():
    """Get the local backends."""
    return simulators._localsimulator.local_backends()

def remote_backends(api):
    """Get the remote backends.

    Queries network API if it exists and gets the backends that are online.

    Returns:
        List of online backends if the online api has been set or an empty
        list of it has not been set.
    """
    return [backend['name'] for backend in api.available_backends() ]

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
        self._local_backends = local_backends()
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
            # Check for the existance of the backend
            for qj in q_jobs:
                if qj.backend not in self._online_backends + self._local_backends:
                    raise QISKitError("Backend %s not found!" % qj.backend)

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
            wait (int): Wait time is how long to check if all jobs is completed
            timeout (int): Time waiting for the results
            silent (bool): If true, prints out results
        """
        executor = self.executor_class(max_workers=self.max_workers)
        for q_job in self.q_jobs:
            if q_job.backend in self._local_backends:
                future = executor.submit(run_local_simulator,
                                         q_job.qobj)
            elif self.online and q_job.backend in self._online_backends:
                future = executor.submit(run_remote_backend,
                                         q_job.qobj,
                                         self._api, wait=wait, timeout=timeout,
                                         silent=silent)
            future.silent = silent
            future.qobj = q_job.qobj
            future.add_done_callback(self._job_done_callback)
            self.futures[future] = q_job.qobj
