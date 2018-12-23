


The IBM Q provider
==================

In Qiskit we have an interface for backends and jobs that will be useful
for running circuits and extending to third-party backends. In this
tutorial, we will review the core components of Qiskit’s base backend
framework, using the IBM Q provider as an example.

The interface has three parts: the provider, the backend, and the job:

-  provider: accesses backends and provides backend objects
-  backend: runs the quantum circuit
-  job: keeps track of the submitted job

The Provider
------------

The IBMQ Provider is an entity that provides access to a group of
different backends (for example, backends available through IBM Q
Experience or IBM Q Network).

The IBMQ provider inherits from BaseProvider and implements the methods:

-  ``backends()``: returns all backend objects known to the provider.
-  ``get_backend(name)``: returns the named backend.

The IBM Q provider has some extra functions for handling administrative
tasks. The credentials can be saved to disk or used in a session and
never saved.

-  ``enable_account(token, url)``: enable an account in the current
   session
-  ``disable_accounts(**kwargs)``: disable one or more accounts from
   current session
-  ``save_account(token, url)``: save an account to disk
-  ``delete_accounts(**kwargs)``: delete the account or accounts from
   disk
-  ``load_accounts(**kwargs)``: load previously-saved account or
   accounts into session
-  ``active_accounts()``: list all accounts active in this session
-  ``stored_accounts()``: list all accounts saved to disk

.. code:: ipython3

    from qiskit import IBMQ 
    
    IBMQ.backends()




.. parsed-literal::

    []



Here we see that there are no backends. This is because no accounts have
been loaded.

Let’s start fresh and delete any accounts on disk. If no accounts are on
disk this will error

.. code:: ipython3

    IBMQ.delete_accounts()

verify that there are no accounts stored now

.. code:: ipython3

    IBMQ.stored_accounts()




.. parsed-literal::

    []



To demonstrate that we can load multiple accounts using the IBMQ
provider, here we use two files ``Qconfig_IBMQ_experience.py`` and
``Qconfig_IBMQ_network.py``, which are just containers of the
``APItoken`` and ``URL``.

-  ``APItoken = 'MY_API_TOKEN'``
-  ``URL = 'THE_URL'``

For the IBM Q experience the URL is not needed and is loaded by default
in ``enable_account`` and ``save_account``. For the IBM Q Network the
url is found on your q-console account page. We don’t recommend saving
and using files like this. We recommend just inputting the ``APItoken``
and ``URL`` directly into the methods ``enable_account`` and
``save_account``.

.. code:: ipython3

    import Qconfig_IBMQ_experience
    import Qconfig_IBMQ_network

To enable an account (useful for one-off use, or if you don’t want to
save to disk)

.. code:: ipython3

    IBMQ.enable_account(Qconfig_IBMQ_experience.APItoken)

To see that accounts which are enabled for use

.. code:: ipython3

    # uncomment to print to screen (it will show your token and url)
    # IBMQ.active_accounts()

and backends which are available

.. code:: ipython3

    IBMQ.backends()




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>]



Disable that account (so we go back to no accounts active)

.. code:: ipython3

    IBMQ.disable_accounts(token=Qconfig_IBMQ_experience.APItoken)

Now no backends are available

.. code:: ipython3

    IBMQ.backends()




.. parsed-literal::

    []



Save two accounts: a public (IBM Q experience) and a premium (IBM Q
network)

.. code:: ipython3

    IBMQ.save_account(Qconfig_IBMQ_experience.APItoken, overwrite=True)
    IBMQ.save_account(Qconfig_IBMQ_network.APItoken, Qconfig_IBMQ_network.url, overwrite=True)

Now they should show up as present on disk

.. code:: ipython3

    # uncomment to print to screen (it will show your token and url)
    # IBMQ.stored_accounts()

but no account active in current session yet

.. code:: ipython3

    IBMQ.active_accounts()




.. parsed-literal::

    []



so IBMQ can’t see any backends yet

.. code:: ipython3

    IBMQ.backends()




.. parsed-literal::

    []



now load up every account stored to disk

.. code:: ipython3

    IBMQ.load_accounts()

backends from two different accounts available for use

.. code:: ipython3

    IBMQ.backends()




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>,
     <IBMQBackend('ibmq_20_tokyo') from IBMQ(ibm-q-internal, research, yorktown)>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ(ibm-q-internal, research, yorktown)>]



now if you want to work with backends of a single account, you can do so
via account filtering

.. code:: ipython3

    IBMQ.backends(hub='ibm-q-internal')




.. parsed-literal::

    [<IBMQBackend('ibmq_20_tokyo') from IBMQ(ibm-q-internal, research, yorktown)>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ(ibm-q-internal, research, yorktown)>]



but you can also just disable account in the current session

.. code:: ipython3

    IBMQ.disable_accounts(hub='ibm-q-internal')

so now only one account is active

.. code:: ipython3

    # uncomment to print to screen (it will show your token and url)
    # IBMQ.active_accounts()

and only that account’s backends are available

.. code:: ipython3

    IBMQ.backends()




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>]



or from the start use the filtering to just load up that account you’re
interested in

.. code:: ipython3

    IBMQ.disable_accounts()
    IBMQ.load_accounts(hub=None)
    IBMQ.backends()




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>]



Filtering the backends
----------------------

You may also optionally filter the set of returned backends, by passing
arguments that query the backend’s ``configuration`` or ``status`` or
``properties``. The filters are passed by conditions and for more
general filters you can make advanced functions using the lambda
function.

As a first example: only return currently operational devices

.. code:: ipython3

    IBMQ.backends(operational=True, simulator=False)




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>]



only return backends that are real devices, have more than 10 qubits and
are operational

.. code:: ipython3

    IBMQ.backends(filters=lambda x: x.configuration().n_qubits <= 5 and 
                  not x.configuration().simulator and x.status().operational==True)




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>]



Filter: show the least busy device (in terms of pending jobs in the
queue)

.. code:: ipython3

    from qiskit.providers.ibmq import least_busy
    
    small_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits == 5 and
                                                           not x.configuration().simulator)
    least_busy(small_devices)




.. parsed-literal::

    <IBMQBackend('ibmqx4') from IBMQ()>



The above filters can be combined as desired.

If you just want to get an instance of a particular backend, you can use
the ``get_backend()`` method.

.. code:: ipython3

    IBMQ.get_backend('ibmq_16_melbourne')




.. parsed-literal::

    <IBMQBackend('ibmq_16_melbourne') from IBMQ()>



The backend
-----------

Backends represent either a simulator or a real quantum computer, and
are responsible for running quantum circuits and returning results. They
have a ``run`` method which takes in a ``qobj`` as input, which is a
quantum object and the result of the compilation process, and returns a
BaseJob object. This object allows asynchronous running of jobs for
retrieving results from a backend when the job is completed.

At a minimum, backends use the following methods, inherited from
BaseBackend:

-  ``provider`` - returns the provider of the backend
-  ``name()`` - gets the name of the backend.
-  ``status()`` - gets the status of the backend.
-  ``configuration()`` - gets the configuration of the backend.
-  ``properties()`` - gets the properties of the backend.
-  ``run()`` - runs a qobj on the backend.

For remote backends they must support the additional

-  ``jobs()`` - returns a list of previous jobs executed by this user on
   this backend.
-  ``retrieve_job()`` - returns a job by a job_id.

In future updates they will introduce the following commands

-  ``defaults()`` - gives a data structure of typical default
   parameters.
-  ``schema()`` - gets a schema for the backend

There are some IBMQ only functions

-  ``hub`` - returns the IBMQ hub for this backend.
-  ``group`` - returns the IBMQ group for this backend.
-  ``project`` - returns the IBMQ project for this backend.

.. code:: ipython3

    backend = least_busy(small_devices)

Let’s start with the ``backend.provider``, which returns a provider
object

.. code:: ipython3

    backend.provider




.. parsed-literal::

    <bound method BaseBackend.provider of <IBMQBackend('ibmqx4') from IBMQ()>>



Next is the ``name()``, which returns the name of the backend

.. code:: ipython3

    backend.name()




.. parsed-literal::

    'ibmqx4'



Next let’s look at the ``status()``:

::

   operational lets you know that the backend is taking jobs
   pending_jobs lets you know how many jobs are in the queue

.. code:: ipython3

    backend.status()




.. parsed-literal::

    BackendStatus(backend_name='ibmqx4', backend_version='1.0.0', operational=True, pending_jobs=0, status_msg='active')



The next is ``configuration()``

.. code:: ipython3

    backend.configuration()




.. parsed-literal::

    BackendConfiguration(allow_q_object=True, backend_name='ibmqx4', backend_version='1.0.0', basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], conditional=False, coupling_map=[[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]], credits_required=True, description='5 qubit device', gates=[GateConfig(coupling_map=[[0], [1], [2], [3], [4]], name='id', parameters=[], qasm_def='gate id q { U(0,0,0) q; }'), GateConfig(coupling_map=[[0], [1], [2], [3], [4]], name='u1', parameters=['lambda'], qasm_def='gate u1(lambda) q { U(0,0,lambda) q; }'), GateConfig(coupling_map=[[0], [1], [2], [3], [4]], name='u2', parameters=['phi', 'lambda'], qasm_def='gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'), GateConfig(coupling_map=[[0], [1], [2], [3], [4]], name='u3', parameters=['theta', 'phi', 'lambda'], qasm_def='u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'), GateConfig(coupling_map=[[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]], name='cx', parameters=[], qasm_def='gate cx q1,q2 { CX q1,q2; }')], local=False, max_experiments=75, max_shots=8192, memory=True, n_qubits=5, n_registers=1, online_date=datetime.datetime(2018, 11, 6, 5, 0, tzinfo=tzutc()), open_pulse=False, sample_name='raven', simulator=False, url='None')



The next is ``properties()`` method

.. code:: ipython3

    backend.properties()




.. parsed-literal::

    BackendProperties(backend_name='ibmqx4', backend_version='1.0.0', gates=[Gate(gate='u1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0)], qubits=[0]), Gate(gate='u2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0007726307293453583)], qubits=[0]), Gate(gate='u3', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0015452614586907165)], qubits=[0]), Gate(gate='u1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0)], qubits=[1]), Gate(gate='u2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.00197489316929661)], qubits=[1]), Gate(gate='u3', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.00394978633859322)], qubits=[1]), Gate(gate='u1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0)], qubits=[2]), Gate(gate='u2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.001631340796924452)], qubits=[2]), Gate(gate='u3', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.003262681593848904)], qubits=[2]), Gate(gate='u1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0)], qubits=[3]), Gate(gate='u2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.001889001411209068)], qubits=[3]), Gate(gate='u3', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.003778002822418136)], qubits=[3]), Gate(gate='u1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0)], qubits=[4]), Gate(gate='u2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.0033494941004675316)], qubits=[4]), Gate(gate='u3', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 10, 56, 45, tzinfo=tzutc()), name='gate_error', unit='', value=0.006698988200935063)], qubits=[4]), Gate(gate='cx', name='CX1_0', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 24, 19, tzinfo=tzutc()), name='gate_error', unit='', value=0.03638715304639503)], qubits=[1, 0]), Gate(gate='cx', name='CX2_0', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 30, 8, tzinfo=tzutc()), name='gate_error', unit='', value=0.0260837887197298)], qubits=[2, 0]), Gate(gate='cx', name='CX2_1', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 35, 38, tzinfo=tzutc()), name='gate_error', unit='', value=0.040748317062039324)], qubits=[2, 1]), Gate(gate='cx', name='CX3_2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 40, 54, tzinfo=tzutc()), name='gate_error', unit='', value=0.06022428067792304)], qubits=[3, 2]), Gate(gate='cx', name='CX3_4', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 47, 54, tzinfo=tzutc()), name='gate_error', unit='', value=0.04131394123324597)], qubits=[3, 4]), Gate(gate='cx', name='CX4_2', parameters=[Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='gate_error', unit='', value=0.061264181329610395)], qubits=[4, 2])], general=[], last_update_date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), qubits=[[Nduv(date=datetime.datetime(2018, 12, 19, 2, 14, 12, tzinfo=tzutc()), name='T1', unit='µs', value=52.877964468812685), Nduv(date=datetime.datetime(2018, 12, 19, 2, 15, 13, tzinfo=tzutc()), name='T2', unit='µs', value=45.91461986614799), Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='frequency', unit='GHz', value=5.249846359615167), Nduv(date=datetime.datetime(2018, 12, 19, 2, 13, 55, tzinfo=tzutc()), name='readout_error', unit='', value=0.060249999999999915)], [Nduv(date=datetime.datetime(2018, 12, 19, 2, 14, 12, tzinfo=tzutc()), name='T1', unit='µs', value=52.189109032554136), Nduv(date=datetime.datetime(2018, 12, 19, 2, 15, 56, tzinfo=tzutc()), name='T2', unit='µs', value=19.451959460737445), Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='frequency', unit='GHz', value=5.295776909561718), Nduv(date=datetime.datetime(2018, 12, 19, 2, 13, 55, tzinfo=tzutc()), name='readout_error', unit='', value=0.07424999999999993)], [Nduv(date=datetime.datetime(2018, 12, 19, 2, 14, 12, tzinfo=tzutc()), name='T1', unit='µs', value=42.880247006778106), Nduv(date=datetime.datetime(2018, 12, 19, 2, 16, 37, tzinfo=tzutc()), name='T2', unit='µs', value=29.48085688756878), Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='frequency', unit='GHz', value=5.353246798006777), Nduv(date=datetime.datetime(2018, 12, 19, 2, 13, 55, tzinfo=tzutc()), name='readout_error', unit='', value=0.029249999999999998)], [Nduv(date=datetime.datetime(2018, 12, 19, 2, 14, 12, tzinfo=tzutc()), name='T1', unit='µs', value=46.880348727946355), Nduv(date=datetime.datetime(2018, 12, 19, 2, 15, 56, tzinfo=tzutc()), name='T2', unit='µs', value=17.744486787296733), Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='frequency', unit='GHz', value=5.434943769576225), Nduv(date=datetime.datetime(2018, 12, 19, 2, 13, 55, tzinfo=tzutc()), name='readout_error', unit='', value=0.02300000000000002)], [Nduv(date=datetime.datetime(2018, 12, 19, 2, 14, 12, tzinfo=tzutc()), name='T1', unit='µs', value=41.224715178255046), Nduv(date=datetime.datetime(2018, 12, 19, 2, 15, 13, tzinfo=tzutc()), name='T2', unit='µs', value=11.096548052083062), Nduv(date=datetime.datetime(2018, 12, 19, 2, 53, 6, tzinfo=tzutc()), name='frequency', unit='GHz', value=5.175820586522991), Nduv(date=datetime.datetime(2018, 12, 19, 2, 13, 55, tzinfo=tzutc()), name='readout_error', unit='', value=0.07525000000000004)]])



The next is ``hub``, ``group``, and ``project``. For the IBM Q
experience these will return ``None``

.. code:: ipython3

    backend.hub

.. code:: ipython3

    backend.group

.. code:: ipython3

    backend.project

To see your last 5 jobs ran on the backend use the ``jobs()`` method of
that backend

.. code:: ipython3

    for ran_job in backend.jobs(limit=5):
        print(str(ran_job.job_id()) + " " + str(ran_job.status()))


.. parsed-literal::

    5c1a2b4f39c21300575b61b0 JobStatus.DONE
    5c1a2b2439c21300575b61ae JobStatus.DONE
    5c19dd832065d5005c4bd8fb JobStatus.DONE
    5c19dd79ed804c0056195e67 JobStatus.DONE
    5c19dd784051c50054922e49 JobStatus.DONE


Then the job can be retreived using ``retrieve_job(job_id())`` method

.. code:: ipython3

    job = backend.retrieve_job(ran_job.job_id())

The Job object
--------------

Job instances can be thought of as the “ticket” for a submitted job.
They find out the execution’s state at a given point in time (for
example, if the job is queued, running, or has failed) and also allow
control over the job. They have the following methods:

-  ``status()`` - returns the status of the job.
-  ``backend()`` - returns the backend the job was run on.
-  ``job_id()`` - gets the job_id.
-  ``cancel()`` - cancels the job.
-  ``result()`` - gets the results from the circuit run.

IBMQ only functions

-  ``creation_date()`` - gives the date at which the job was created.
-  ``queue_position()`` - gives the position of the job in the queue.
-  ``error_message()`` - gives the error message of failed jobs.

Let’s start with the ``status()``. This returns the job status and a
message

.. code:: ipython3

    job.status()




.. parsed-literal::

    <JobStatus.DONE: 'job has successfully run'>



To get a backend object from the job use the ``backend()`` method

.. code:: ipython3

    backend_temp = job.backend()
    backend_temp




.. parsed-literal::

    <IBMQBackend('ibmqx4') from IBMQ()>



To get the job_id use the ``job_id()`` method

.. code:: ipython3

    job.job_id()




.. parsed-literal::

    '5c19dd784051c50054922e49'



To get the result from the job use the ``result()`` method

.. code:: ipython3

    result = job.result()
    counts = result.get_counts()
    print(counts)


.. parsed-literal::

    {'00': 440, '10': 49, '01': 75, '11': 460}


If you want to check the creation date use ``creation_date()``

.. code:: ipython3

    job.creation_date()




.. parsed-literal::

    '2018-12-19T05:56:08.934Z'



Let’s make an active example

.. code:: ipython3

    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import compile

.. code:: ipython3

    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.cx(qr[0], qr[1])
    circuit.measure(qr, cr)




.. parsed-literal::

    <qiskit.circuit.instructionset.InstructionSet at 0xa167e94a8>



To compile this circuit for the backend use the compile function. It
will make a qobj (quantum object) that can be run on the backend using
the ``run(qobj)`` method.

.. code:: ipython3

    qobj = compile(circuit, backend=backend, shots=1024)
    job = backend.run(qobj)

The status of this job can be checked with the ``status()`` method

.. code:: ipython3

    job.status()




.. parsed-literal::

    <JobStatus.INITIALIZING: 'job is being initialized'>



If you made a mistake and need to cancel the job use the ``cancel()``
method.

.. code:: ipython3

    import time
    #time.sleep(10)
    
    job.cancel()




.. parsed-literal::

    False



The ``status()`` will show that the job cancelled.

.. code:: ipython3

    job.status()




.. parsed-literal::

    <JobStatus.QUEUED: 'job is queued'>



To rerun the job and set up a loop to check the status and queue
position you can use the ``queue_position()`` method.

.. code:: ipython3

    job = backend.run(qobj)

.. code:: ipython3

    from qiskit.tools.monitor import job_monitor
    job_monitor(job)
    result = job.result()



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


.. code:: ipython3

    counts = result.get_counts()
    print(counts)


.. parsed-literal::

    {'111': 46, '100': 58, '011': 99, '101': 504, '110': 18, '000': 29, '010': 35, '001': 235}

