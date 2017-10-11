import time
from qiskit.backends._basebackend import BaseBackend
from qiskit.backends._backendutils import get_backend_configuration
from qiskit import _openquantumcompiler as openquantumcompiler
from qiskit._result import Result

from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience

class QeRemote(BaseBackend):
    def __init__(self, configuration=None):
        """Initialize remote backend for IBM Quantum Experience.

        Args:

        """
        if configuration is None:
            self._configuration = get_backend_configuration(backend_name)
        else:
            self._configuration = configuration
        self._configuration['local'] = False

    def run(self, q_job):
        """Run jobs

        Args:
            q_job (QuantumJob): job to run

        Returns:
            Result object.

        Raises:
            ResultError: if the api put 'error' in its output
        """
        self._qobj = q_job.qobj
        wait = q_job.wait
        timeout = q_job.timeout
        silent = q_job.silent
        api_jobs = []
        for circuit in self._qobj['circuits']:
            if (('compiled_circuit_qasm' not in circuit) or
                    (circuit['compiled_circuit_qasm'] is None)):
                compiled_circuit = openquantumcompiler.compile(
                    circuit['circuit'].qasm())
                circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)
            if isinstance(circuit['compiled_circuit_qasm'], bytes):
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm'].decode()})
            else:
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm']})

        seed0 = self._qobj['circuits'][0]['config']['seed']
        output = self._api.run_job(api_jobs, self._qobj['config']['backend'],
                             shots=self._qobj['config']['shots'],
                             max_credits=self._qobj['config']['max_credits'],
                             seed=seed0)
        if 'error' in output:
            raise ResultError(output['error'])

        job_result = _wait_for_job(output['id'], self._api, wait=wait,
                                   timeout=timeout, silent=silent)
        job_result['name'] = self._qobj['id']
        job_result['backend'] = self._qobj['config']['backend']
        this_result = Result(job_result, self._qobj)
        return this_result

def _wait_for_job(jobid, api, wait=5, timeout=60):
    """Wait until all online ran circuits of a qobj are 'COMPLETED'.

    Args:
        jobid:  is a list of id strings.
        api (IBMQuantumExperience): IBMQuantumExperience API connection
        wait (int):  is the time to wait between requests, in seconds
        timeout (int):  is how long we wait before failing, in seconds

    Returns:
        A list of results that correspond to the jobids.

    Raises:
        QISKitError:
    """
    timer = 0
    job_result = api.get_job(jobid)
    if 'status' not in job_result:
        raise QISKitError("get_job didn't return status: %s" %
                          (pprint.pformat(job_result)))

    while job_result['status'] == 'RUNNING':
        if timer >= timeout:
            return {'status': 'ERROR', 'result': 'Time Out'}
        time.sleep(wait)
        timer += wait
        logger.info('status = %s (%d seconds)', job_result['status'], timer)
        job_result = api.get_job(jobid)

        if 'status' not in job_result:
            raise QISKitError("get_job didn't return status: %s" %
                              (pprint.pformat(job_result)))
        if (job_result['status'] == 'ERROR_CREATING_JOB' or
                job_result['status'] == 'ERROR_RUNNING_JOB'):
            return {'status': 'ERROR', 'result': job_result['status']}

    # Get the results
    job_result_return = []
    for index in range(len(job_result['qasms'])):
        job_result_return.append({'data': job_result['qasms'][index]['data'],
                                  'status': job_result['qasms'][index]['status']})
    return {'status': job_result['status'], 'result': job_result_return}
