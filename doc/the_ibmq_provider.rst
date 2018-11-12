


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




.. code-block:: text

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




.. code-block:: text

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

    import Qconfig_IBMQ_network
    import Qconfig_IBMQ_experience

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




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmqx5') from IBMQ()>,
     <IBMQBackend('ibmqx2') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>]



Disable that account (so we go back to no accounts active)

.. code:: ipython3

    IBMQ.disable_accounts(token=Qconfig_IBMQ_experience.APItoken)

Now no backends are available

.. code:: ipython3

    IBMQ.backends()




.. code-block:: text

    []



Save two accounts: a public (IBM Q experience) and a premium (IBM Q
network)

.. code:: ipython3

    IBMQ.save_account(Qconfig_IBMQ_experience.APItoken)
    IBMQ.save_account(Qconfig_IBMQ_network.APItoken, Qconfig_IBMQ_network.url)

Now they should show up as present on disk

.. code:: ipython3

    # uncomment to print to screen (it will show your token and url)
    # IBMQ.stored_accounts()

but no account active in current session yet

.. code:: ipython3

    IBMQ.active_accounts()




.. code-block:: text

    []



so IBMQ can’t see any backends yet

.. code:: ipython3

    IBMQ.backends()




.. code-block:: text

    []



now load up every account stored to disk

.. code:: ipython3

    IBMQ.load_accounts()

backends from two different accounts available for use

.. code:: ipython3

    IBMQ.backends()




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmqx5') from IBMQ()>,
     <IBMQBackend('ibmqx2') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>,
     <IBMQBackend('ibmq_20_tokyo') from IBMQ(ibm-q-internal, qiskit, qiskit-terra)>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ(ibm-q-internal, qiskit, qiskit-terra)>]



now if you want to work with backends of a single account, you can do so
via account filtering

.. code:: ipython3

    IBMQ.backends(hub='ibm-q-internal')




.. code-block:: text

    [<IBMQBackend('ibmq_20_tokyo') from IBMQ(ibm-q-internal, qiskit, qiskit-terra)>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ(ibm-q-internal, qiskit, qiskit-terra)>]



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




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmqx5') from IBMQ()>,
     <IBMQBackend('ibmqx2') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQBackend('ibmq_qasm_simulator') from IBMQ()>]



or from the start use the filtering to just load up that account you’re
interested in

.. code:: ipython3

    IBMQ.disable_accounts()
    IBMQ.load_accounts(hub=None)
    IBMQ.backends()




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmqx5') from IBMQ()>,
     <IBMQBackend('ibmqx2') from IBMQ()>,
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




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>]



only return backends that are real devices, have more than 10 qubits and
are operational

.. code:: ipython3

    IBMQ.backends(filters=lambda x: x.configuration()['n_qubits'] <= 5 and 
                  not x.configuration()['simulator'] and x.status()['operational']==True)




.. code-block:: text

    [<IBMQBackend('ibmqx4') from IBMQ()>]



Filter: show the least busy device (in terms of pending jobs in the
queue)

.. code:: ipython3

    from qiskit.backends.ibmq import least_busy
    
    small_devices = IBMQ.backends(filters=lambda x: x.configuration()['n_qubits'] <= 5 and
                                                           not x.configuration()['simulator'])
    least_busy(small_devices)




.. code-block:: text

    <IBMQBackend('ibmqx4') from IBMQ()>



The above filters can be combined as desired.

If you just want to get an instance of a particular backend, you can use
the ``get_backend()`` method.

.. code:: ipython3

    IBMQ.get_backend('ibmq_16_melbourne')




.. code-block:: text

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




.. code-block:: text

    <qiskit.backends.ibmq.ibmqprovider.IBMQProvider at 0x10e62cfd0>



Next is the ``name()``, which returns the name of the backend

.. code:: ipython3

    backend.name()




.. code-block:: text

    'ibmqx4'



Next let’s look at the ``status()``:

::

   operational lets you know that the backend is taking jobs
   pending_jobs lets you know how many jobs are in the queue

.. code:: ipython3

    backend.status()




.. code-block:: text

    {'pending_jobs': 6, 'name': 'ibmqx4', 'operational': True}



The next is ``configuration()``

.. code:: ipython3

    backend.configuration()




.. code-block:: text

    {'local': False,
     'name': 'ibmqx4',
     'version': '1.2.0',
     'description': '5 qubit transmon bowtie chip 3',
     'gate_set': 'SU2+CNOT',
     'basis_gates': 'u1,u2,u3,cx,id',
     'online_date': '2017-09-18T00:00:00.000Z',
     'chip_name': 'Raven',
     'deleted': False,
     'url': 'https://ibm.biz/qiskit-ibmqx4',
     'internal_id': '5ae875670f020500393162b3',
     'simulator': False,
     'allow_q_object': False,
     'n_qubits': 5,
     'coupling_map': [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]}



The next is ``properties()`` method

.. code:: ipython3

    backend.properties()




.. code-block:: text

    {'last_update_date': '2018-11-12T02:56:39.000Z',
     'qubits': [{'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.0011160854878761173},
       'name': 'Q0',
       'readoutError': {'date': '2018-11-12T02:56:39Z', 'value': 0.073},
       'buffer': {'date': '2018-11-12T02:56:39Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-12T02:56:39Z', 'value': 60, 'unit': 'ns'},
       'T2': {'date': '2018-11-12T02:56:39Z', 'value': 32.8, 'unit': 'µs'},
       'T1': {'date': '2018-11-12T02:56:39Z', 'value': 44.6, 'unit': 'µs'},
       'frequency': {'date': '2018-11-12T02:56:39Z',
        'value': 5.24987,
        'unit': 'GHz'}},
      {'gateError': {'date': '2018-11-12T02:56:39Z', 'value': 0.00128782749692391},
       'name': 'Q1',
       'readoutError': {'date': '2018-11-12T02:56:39Z', 'value': 0.073},
       'buffer': {'date': '2018-11-12T02:56:39Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-12T02:56:39Z', 'value': 60, 'unit': 'ns'},
       'T2': {'date': '2018-11-12T02:56:39Z', 'value': 20.1, 'unit': 'µs'},
       'T1': {'date': '2018-11-12T02:56:39Z', 'value': 34.2, 'unit': 'µs'},
       'frequency': {'date': '2018-11-12T02:56:39Z',
        'value': 5.29577,
        'unit': 'GHz'}},
      {'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.001631340796924452},
       'name': 'Q2',
       'readoutError': {'date': '2018-11-12T02:56:39Z', 'value': 0.033},
       'buffer': {'date': '2018-11-12T02:56:39Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-12T02:56:39Z', 'value': 60, 'unit': 'ns'},
       'T2': {'date': '2018-11-12T02:56:39Z', 'value': 27.4, 'unit': 'µs'},
       'T1': {'date': '2018-11-12T02:56:39Z', 'value': 38, 'unit': 'µs'},
       'frequency': {'date': '2018-11-12T02:56:39Z',
        'value': 5.35326,
        'unit': 'GHz'}},
      {'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.002232583111384412},
       'name': 'Q3',
       'readoutError': {'date': '2018-11-12T02:56:39Z', 'value': 0.026},
       'buffer': {'date': '2018-11-12T02:56:39Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-12T02:56:39Z', 'value': 60, 'unit': 'ns'},
       'T2': {'date': '2018-11-12T02:56:39Z', 'value': 12.4, 'unit': 'µs'},
       'T1': {'date': '2018-11-12T02:56:39Z', 'value': 41.2, 'unit': 'µs'},
       'frequency': {'date': '2018-11-12T02:56:39Z',
        'value': 5.43497,
        'unit': 'GHz'}},
      {'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.0013737021608475342},
       'name': 'Q4',
       'readoutError': {'date': '2018-11-12T02:56:39Z', 'value': 0.056},
       'buffer': {'date': '2018-11-12T02:56:39Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-12T02:56:39Z', 'value': 60, 'unit': 'ns'},
       'T2': {'date': '2018-11-12T02:56:39Z', 'value': 12, 'unit': 'µs'},
       'T1': {'date': '2018-11-12T02:56:39Z', 'value': 50.3, 'unit': 'µs'},
       'frequency': {'date': '2018-11-12T02:56:39Z',
        'value': 5.17582,
        'unit': 'GHz'}}],
     'multi_qubit_gates': [{'qubits': [1, 0],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.039730922592824625},
       'name': 'CX1_0'},
      {'qubits': [2, 0],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z', 'value': 0.0370616990430509},
       'name': 'CX2_0'},
      {'qubits': [2, 1],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z',
        'value': 0.039182981129817634},
       'name': 'CX2_1'},
      {'qubits': [3, 2],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z', 'value': 0.06468197586341454},
       'name': 'CX3_2'},
      {'qubits': [3, 4],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z', 'value': 0.0472178292725369},
       'name': 'CX3_4'},
      {'qubits': [4, 2],
       'type': 'CX',
       'gateError': {'date': '2018-11-12T02:56:39Z', 'value': 0.06971263047107376},
       'name': 'CX4_2'}],
     'backend': 'ibmqx4',
     'fridge_parameters': {'cooldownDate': '2017-09-07',
      'Temperature': {'date': '-', 'value': [], 'unit': '-'}}}



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


.. code-block:: text

    5be8f39c54dacb0059c2b0db JobStatus.DONE
    5be8f394afd471005540d4d7 JobStatus.CANCELLED
    5be8ae5e17436b0052751909 JobStatus.DONE
    5be748a7e00f60005ad7f23d JobStatus.DONE
    5be746e3d4d36f0054595d60 JobStatus.DONE


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




.. code-block:: text

    <JobStatus.DONE: 'job has successfully run'>



To get a backend object from the job use the ``backend()`` method

.. code:: ipython3

    backend_temp = job.backend()
    backend_temp




.. code-block:: text

    <IBMQBackend('ibmqx4') from IBMQ()>



To get the job_id use the ``job_id()`` method

.. code:: ipython3

    job.job_id()




.. code-block:: text

    '5be746e3d4d36f0054595d60'



To get the result from the job use the ``result()`` method

.. code:: ipython3

    result = job.result()
    counts = result.get_counts()
    print(counts)


.. code-block:: text

    {'000': 387, '001': 23, '010': 39, '011': 22, '100': 30, '101': 59, '110': 62, '111': 402}


If you want to check the creation date use ``creation_date()``

.. code:: ipython3

    job.creation_date()




.. code-block:: text

    '2018-11-10T21:00:19.795Z'



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




.. code-block:: text

    <qiskit._instructionset.InstructionSet at 0x10ffb5240>



To compile this circuit for the backend use the compile function. It
will make a qobj (quantum object) that can be run on the backend using
the ``run(qobj)`` method.

.. code:: ipython3

    qobj = compile(circuit, backend=backend, shots=1024)
    job = backend.run(qobj)

The status of this job can be checked with the ``status()`` method

.. code:: ipython3

    job.status()




.. code-block:: text

    <JobStatus.QUEUED: 'job is queued'>



If you made a mistake and need to cancel the job use the ``cancel()``
method.

.. code:: ipython3

    import time
    #time.sleep(10)
    
    job.cancel()




.. code-block:: text

    True



The ``status()`` will show that the job cancelled.

.. code:: ipython3

    job.status()




.. code-block:: text

    <JobStatus.CANCELLED: 'job has been cancelled'>



To rerun the job and set up a loop to check the status and queue
position you can use the ``queue_position()`` method.

.. code:: ipython3

    job = backend.run(qobj)

.. code:: ipython3

    lapse = 0
    interval = 60
    while job.status().name != 'DONE':
        print('Status @ {} seconds'.format(interval * lapse))
        print(job.status())
        print(job.queue_position())
        time.sleep(interval)
        lapse += 1
    print(job.status())
    result = job.result()


.. code-block:: text

    Status @ 0 seconds
    JobStatus.INITIALIZING
    None
    JobStatus.DONE


.. code:: ipython3

    counts = result.get_counts()
    print(counts)


.. code-block:: text

    {'000': 37, '001': 155, '010': 55, '011': 50, '100': 86, '101': 582, '110': 20, '111': 39}

