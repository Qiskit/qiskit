


The IBMQ provider
=================

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




.. parsed-literal::

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




.. parsed-literal::

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




.. parsed-literal::

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




.. parsed-literal::

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




.. parsed-literal::

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




.. parsed-literal::

    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>]



only return backends that are real devices, have more than 10 qubits and
are operational

.. code:: ipython3

    IBMQ.backends(filters=lambda x: x.configuration()['n_qubits'] > 10 and 
                  not x.configuration()['simulator'] and x.status()['operational']==True)




.. parsed-literal::

    [<IBMQBackend('ibmq_16_melbourne') from IBMQ()>]



Filter: show the least busy device (in terms of pending jobs in the
queue)

.. code:: ipython3

    from qiskit.backends.ibmq import least_busy
    least_busy(IBMQ.backends(simulator=False))




.. parsed-literal::

    <IBMQBackend('ibmq_16_melbourne') from IBMQ()>



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

    backend = least_busy(IBMQ.backends(simulator=False))

Let’s start with the ``backend.provider``, which returns a provider
object

.. code:: ipython3

    backend.provider




.. parsed-literal::

    <qiskit.backends.ibmq.ibmqprovider.IBMQProvider at 0x118cfbfd0>



Next is the ``name()``, which returns the name of the backend

.. code:: ipython3

    backend.name()




.. parsed-literal::

    'ibmq_16_melbourne'



Next let’s look at the ``status()``:

::

   operational lets you know that the backend is taking jobs
   pending_jobs lets you know how many jobs are in the queue

.. code:: ipython3

    backend.status()




.. parsed-literal::

    {'pending_jobs': 0, 'name': 'ibmq_16_melbourne', 'operational': True}



The next is ``configuration()``

.. code:: ipython3

    backend.configuration()




.. parsed-literal::

    {'local': False,
     'name': 'ibmq_16_melbourne',
     'version': '1.0.0',
     'description': '16 transmon 2x8 ladder',
     'basis_gates': 'u1,u2,u3,cx,id',
     'online_date': '2018-09-07T00:00:00.000Z',
     'chip_name': 'Albatross',
     'deleted': False,
     'url': 'https://ibm.biz/qiskit-ibmq_16_melbourne',
     'internal_id': '5ba502d0986f16003ea56c87',
     'simulator': False,
     'allow_q_object': False,
     'n_qubits': 14,
     'coupling_map': [[1, 0],
      [1, 2],
      [2, 3],
      [4, 3],
      [4, 10],
      [5, 4],
      [5, 6],
      [5, 9],
      [6, 8],
      [7, 8],
      [9, 8],
      [9, 10],
      [11, 3],
      [11, 10],
      [11, 12],
      [12, 2],
      [13, 1],
      [13, 12]]}



The next is ``properties()`` method

.. code:: ipython3

    backend.properties()




.. parsed-literal::

    {'last_update_date': '2018-11-10T07:43:39.000Z',
     'qubits': [{'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0021111108471550954},
       'name': 'Q0',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.03469999999999995},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 15.7, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 59.1, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.1000637}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.006152100589040643},
       'name': 'Q1',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.03699999999999992},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 65.7, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 53.2, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.2383452}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.003740272134946432},
       'name': 'Q2',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.03469999999999995},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 102.6, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 65.7, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.0328719}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0028833222344865628},
       'name': 'Q3',
       'readoutError': {'date': '2018-11-10T07:41:33Z', 'value': 0.1572},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 70, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 56.5, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.8961435}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0019122042304199338},
       'name': 'Q4',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.052100000000000035},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 31.5, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 62.1, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.0261587}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0025980597194751875},
       'name': 'Q5',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.04180000000000006},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 26.2, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 23.4, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.0670346}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.016121168647298623},
       'name': 'Q6',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.43810000000000004},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 0.2, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 47.9, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.9237102}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0021072160137580176},
       'name': 'Q7',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.35119999999999996},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 50.4, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 44.8, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.9744298}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.002027642041947275},
       'name': 'Q8',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.032399999999999984},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 84.3, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 51.3, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.7381231}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0027701731296251864},
       'name': 'Q9',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.027800000000000047},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 74.7, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 43, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.9633109}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0017276307654423562},
       'name': 'Q10',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.05289999999999995},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 72, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 53.3, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.9449706}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.0019387530324384006},
       'name': 'Q11',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.12830000000000008},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 121.4, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 69.7, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 5.0046508}},
      {'gateError': {'date': '2018-11-10T07:47:53Z',
        'value': 0.005277351743300962},
       'name': 'Q12',
       'readoutError': {'date': '2018-11-10T07:41:33Z',
        'value': 0.03770000000000007},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 107.9, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 73.4, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.7598326}},
      {'gateError': {'date': '2018-11-10T07:47:53Z', 'value': 0.00706203056750071},
       'name': 'Q13',
       'readoutError': {'date': '2018-11-10T07:41:33Z', 'value': 0.0655},
       'buffer': {'date': '2018-11-10T07:06:52Z', 'value': 10, 'unit': 'ns'},
       'gateTime': {'date': '2018-11-10T07:06:52Z', 'value': 100, 'unit': 'ns'},
       'T2': {'date': '2018-11-10T07:43:39Z', 'value': 44.1, 'unit': 'µs'},
       'T1': {'date': '2018-11-10T07:41:57Z', 'value': 29.2, 'unit': 'µs'},
       'frequency': {'date': '2018-11-10T07:06:52Z',
        'units': 'GHz',
        'value': 4.9685372}}],
     'multi_qubit_gates': [{'qubits': [1, 0],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z',
        'value': 0.044115602886946104},
       'name': 'CX1_0'},
      {'qubits': [1, 2],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.03071226450213524},
       'name': 'CX1_2'},
      {'qubits': [2, 3],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z',
        'value': 0.052883399201068326},
       'name': 'CX2_3'},
      {'qubits': [4, 3],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.04293561904788096},
       'name': 'CX4_3'},
      {'qubits': [4, 10],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.02567800627055783},
       'name': 'CX4_10'},
      {'qubits': [5, 4],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.04281117233906834},
       'name': 'CX5_4'},
      {'qubits': [5, 6],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.04663507508485301},
       'name': 'CX5_6'},
      {'qubits': [5, 9],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.3278229946644999},
       'name': 'CX5_9'},
      {'qubits': [6, 8],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.3229883820118977},
       'name': 'CX6_8'},
      {'qubits': [7, 8],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.33186235776903733},
       'name': 'CX7_8'},
      {'qubits': [9, 8],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.06028246259895123},
       'name': 'CX9_8'},
      {'qubits': [9, 10],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z',
        'value': 0.038502082712080055},
       'name': 'CX9_10'},
      {'qubits': [11, 10],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.03441489364222472},
       'name': 'CX11_10'},
      {'qubits': [11, 12],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z',
        'value': 0.039247903146692104},
       'name': 'CX11_12'},
      {'qubits': [11, 3],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.0380042285580387},
       'name': 'CX11_3'},
      {'qubits': [12, 2],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.08178999129883763},
       'name': 'CX12_2'},
      {'qubits': [13, 1],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z', 'value': 0.12708741818285327},
       'name': 'CX13_1'},
      {'qubits': [13, 12],
       'type': 'CX',
       'gateError': {'date': '2018-11-10T09:06:09Z',
        'value': 0.029824807923597013},
       'name': 'CX13_12'}],
     'backend': 'ibmq_16_melbourne',
     'fridge_parameters': {'cooldownDate': '2018-07-10',
      'Temperature': {'date': '2018-11-10T10:18:50Z',
       'value': 0.0280663,
       'unit': 'K'}}}



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

    5be8bc0c9a9893006ff69161 JobStatus.CANCELLED
    5be8bbfca9ff0f0053fa28bd JobStatus.DONE
    5be8bbdbb6f6790062d1a7bc JobStatus.DONE
    5be8bb0c6cd471005f3decf2 JobStatus.DONE
    5be8baff9a9893006ff6915f JobStatus.DONE


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

    <IBMQBackend('ibmq_16_melbourne') from IBMQ()>



To get the job_id use the ``job_id()`` method

.. code:: ipython3

    job.job_id()




.. parsed-literal::

    '5be8baff9a9893006ff6915f'



To get the result from the job use the ``result()`` method

.. code:: ipython3

    result = job.result()
    counts = result.get_counts()
    print(counts)


.. parsed-literal::

    {'000': 88, '010': 128, '001': 139, '011': 100, '100': 91, '110': 50, '101': 321, '111': 107}


If you want to check the creation date use ``creation_date()``

.. code:: ipython3

    job.creation_date()




.. parsed-literal::

    '2018-11-11T23:27:59.081Z'



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

    <qiskit._instructionset.InstructionSet at 0x11a6a0320>



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

    True



The ``status()`` will show that the job cancelled.

.. code:: ipython3

    job.status()




.. parsed-literal::

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


.. parsed-literal::

    Status @ 0 seconds
    JobStatus.INITIALIZING
    None
    JobStatus.DONE


.. code:: ipython3

    counts = result.get_counts()
    print(counts)


.. parsed-literal::

    {'000': 88, '010': 114, '001': 155, '011': 90, '100': 130, '110': 40, '101': 347, '111': 60}

