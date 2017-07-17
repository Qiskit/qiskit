"""
Miscellaneous methods.

These are simple methods for common tasks in our examples, including
minor methods for helping with running jobs and getting results.

Author: Andrew Cross, Jay Gambetta
"""
import time
from collections import Counter
import qiskit.unroll as unroll
from qiskit.qasm import Qasm


def get_data(results, i):
    """Get the dict of labels and counts from the output of get_job."""
    return results['qasms'][i]['result']['data']['counts']


def get_job_list_status(jobids, api):
    """Given a list of job ids, return a list of job status.

    jobids is a list of id strings.
    api is an IBMQuantumExperience object.
    """
    status_list = []
    for i in jobids:
        status_list.append(api.get_job(i)['status'])
    return status_list


def run_program(program, api, device, shots, max_credits=3):
    """Run a program (array of quantum circuits).

    program is a list of quantum_circuits
    api the api for the device
    device is a string for real or simulator
    shots is the number of shots
    max_credits is the credits of the experiments.
    """
    jobs = []
    for p in program:
        basis = "u1,u2,u3,cx"  # QE target basis
        u = unroll.Unroller(Qasm(data=p.qasm()).parse(), unroll.CircuitBackend(
                            basis.split(",")))
        C = u.execute()  # returns circuit DAG

        # Do stuff to circuit (not done yet)
        # this is what you send into the API
        jobs.append({'qasm': C.qasm(qeflag=True)})
    out = api.run_job(jobs, device, shots, max_credits)
    return out


def program_to_text(program):
    """Print a program (array of quantum circuits).

    program is a list of quantum_circuits
    """
    jobs = ""
    for p in program:
        basis = "u1,u2,u3,cx"  # QE target basis
        u = unroll.Unroller(Qasm(data=p.qasm()).parse(), unroll.CircuitBackend(
                            basis.split(",")))
        C = u.execute()  # returns circuit DAG
        jobs = jobs + C.qasm(qeflag=True) + "\n\n"
    return jobs


def wait_for_jobs(jobids, api, wait=5, timeout=60):
    """Wait until all status results are 'COMPLETED'.

    jobids is a list of id strings.
    api is an IBMQuantumExperience object.
    wait is the time to wait between requests, in seconds
    timeout is how long we wait before failing, in seconds

    Returns an list of results that correspond to the jobids.
    """
    status = dict(Counter(get_job_list_status(jobids, api)))
    t = 0
    print("status = %s (%d seconds)" % (status, t))
    while 'COMPLETED' not in status or status['COMPLETED'] < len(jobids):
        if t == timeout:
            break
        time.sleep(wait)
        t += wait
        status = dict(Counter(get_job_list_status(jobids, api)))
        print("status = %s (%d seconds)" % (status, t))
    # Get the results
    results = []
    for i in jobids:
        results.append(api.get_job(i))
    return results


def combine_jobs(jobids, api, wait=5, timeout=60):
    """Like wait_for_jobs but with a different return format.

    jobids is a list of id strings.
    api is an IBMQuantumExperience object.
    wait is the time to wait between requests, in seconds
    timeout is how long we wait before failing, in seconds

    Returns a list of dict outcomes of the flattened in the order
    jobids so it works with _getData_.
    """
    results = list(map(lambda x: x['qasms'],
                       wait_for_jobs(jobids, api, wait, timeout)))
    flattened = []
    for sublist in results:
        for val in sublist:
            flattened.append(val)
    # Are there other parts from the dictionary that we want to add,
    # such as shots?
    return {'qasms': flattened}


def average_data(data, observable):
    """Compute the mean value of an observable.

    Takes in the data counts(i) and a corresponding observable in dict
    form and calculates sum_i value(i) P(i) where value(i) is the value of
    the observable for the i state.
    """
    temp = 0
    tot = sum(data.values())
    for key in data:
        if key in observable:
            temp += data[key] * observable[key] / tot
    return temp
