


Jupyter Tools for Terra
=======================

In this section, we will learn how to monitor the status of jobs
submitted to devices and simulators (collectively called backends), as
well as discover how to easily query backend details and view the
collective state of all the backends available to you.

Loading the Monitoring Tools
----------------------------

First, let us load the default qiskit routines, and register our IBMQ
credentials.

.. code:: ipython3

    from qiskit import *
    IBMQ.load_accounts()

Functions for monitoring jobs and backends are here:

.. code:: ipython3

    from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview

If you are running inside a Jupyter notebook, then you will also need to
have ``ipywidgets`` version ``7.3.0`` or higher installed. These come
pre-installed in Anaconda. There are also Jupyter notebook ‘magics’
available for each of the abover functions. The following will register
those magics, making them ready for use.

.. code:: ipython3

    from qiskit.tools.jupyter import *

Tracking Job Status
-------------------

Many times a job(s) submitted to the IBM Q network can take a long time
to process, e.g. jobs with many circuits and/or shots, or may have to
wait in queue for other users. In situations such as these, it is
beneficial to have a way of monitoring the progress of a job, or several
jobs at once. As of Qiskit ``0.6+`` it is possible to monitor the status
of a job in a Jupyter notebook, and also in a Python script (verision
``0.7+``).

Lets see how to make use of these tools.

Monitoring the status of a single job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lets build a simple Bell circuit, submit it to a device, and then
monitor its status.

.. code:: ipython3

    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)
    
    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q, c);

Lets grab the least busy backend

.. code:: ipython3

    from qiskit.providers.ibmq import least_busy
    backend = least_busy(IBMQ.backends(simulator=False))
    backend.name()




.. parsed-literal::

    'ibmqx4'



Monitor the job using ``job_monitor`` in blocking-mode (i.e. using the
same thread as the Python interpretor)

.. code:: ipython3

    job1 = execute(qc, backend)
    job_monitor(job1)



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


Monitor the job using ``job_monitor`` in async-mode (Jupyter notebooks
only). The job will be monitored in a separate thread, allowing you to
continue to work in the notebook.

.. code:: ipython3

    job2 = execute(qc, backend)
    job_monitor(job2, monitor_async=True)



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


It is also possible to monitor the job using the ``qiskit_job_status``
Jupyter notebook magic. This method is always asyncronous.

.. code:: ipython3

    %%qiskit_job_status
    job3 = execute(qc, backend)



.. parsed-literal::

    VBox(children=(HTML(value="<p style='font-size:16px;'>Job Status : job is being initialized </p>"),))


Note that, for the ``qiskit_job_status`` to work, the job returned by
``execute`` must be stored in a variable so that it may be retrieved by
the magic.

Monitoring many jobs simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will monitor many jobs sent the the device. It is if the jobs
are stored in a list to make retrevial easier.

.. code:: ipython3

    num_jobs = 5
    my_jobs = []
    for j in range(num_jobs):
        my_jobs.append(execute(qc, backend))
        job_monitor(my_jobs[j], monitor_async=True)



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


Or, using magic:

.. code:: ipython3

    %%qiskit_job_status
    my_jobs2 = []
    for j in range(num_jobs):
        my_jobs2.append(execute(qc, backend))



.. parsed-literal::

    VBox(children=(HTML(value="<p style='font-size:16px;'>Job Status [0]: job is being initialized </p>"), HTML(va…


In the magics example, the magic is smart enough to know that the list
``my_jobs2`` contains jobs, and will automatically extract them and
check their status. We are not limited to using ``jobs.append()``, and
can use an indexed list or NumPy array as well:

.. code:: ipython3

    %%qiskit_job_status
    import numpy as np
    my_jobs3 = np.empty(num_jobs, dtype=object)
    for j in range(num_jobs):
        my_jobs3[j] = execute(qc, backend)



.. parsed-literal::

    VBox(children=(HTML(value="<p style='font-size:16px;'>Job Status [0]: job is being initialized </p>"), HTML(va…


Changing the interval of status updating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the interval at which the job status is checked is every two
seconds. However, the user is free to change this using the ``interval``
keyword argument in ``job_monitor``

.. code:: ipython3

    job3 = execute(qc, backend)
    job_monitor(job3, interval=5)



.. parsed-literal::

    HTML(value="<p style='font-size:16px;'>Job Status: job is being initialized </p>")


and the ``-i`` or ``--interval`` arguments to the Jupyter magic.

.. code:: ipython3

    %%qiskit_job_status -i 5
    job4 = execute(qc, backend)



.. parsed-literal::

    VBox(children=(HTML(value="<p style='font-size:16px;'>Job Status : job is being initialized </p>"),))


.. code:: ipython3

    %%qiskit_job_status --interval 5
    job5 = execute(qc, backend)



.. parsed-literal::

    VBox(children=(HTML(value="<p style='font-size:16px;'>Job Status : job is being initialized </p>"),))


Backend Details
---------------

So far we have been executing our jobs on a backend, but we have
explored the backends in any detail. For example, we have found the
least busy backend, but do not know if this is the best backend with
respect to gate errors, topology etc. It is possible to get detailed
information for a single backend by calling ``backend_monitor``:

.. code:: ipython3

    backend_monitor(backend)


.. parsed-literal::

    ibmqx4
    ======
    Configuration
    -------------
        n_qubits: 5
        operational: True
        status_msg: active
        pending_jobs: 4
        basis_gates: ['u1', 'u2', 'u3', 'cx', 'id']
        local: False
        simulator: False
        open_pulse: False
        credits_required: True
        conditional: False
        max_experiments: 75
        max_shots: 8192
        coupling_map: [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
        sample_name: raven
        description: 5 qubit device
        n_registers: 1
        memory: True
        url: None
        backend_version: 1.0.0
        backend_name: ibmqx4
        online_date: 2018-11-06T05:00:00+00:00
        allow_q_object: True
    
    Qubits [Name / Freq / T1 / T2 / U1 err / U2 err / U3 err / Readout err]
    -----------------------------------------------------------------------
        Q0 / 5.24985 GHz / 52.87796 µs / 45.91462 µs / 0.0 / 0.00077 / 0.00155 / 0.06025
        Q1 / 5.29578 GHz / 52.18911 µs / 19.45196 µs / 0.0 / 0.00197 / 0.00395 / 0.07425
        Q2 / 5.35325 GHz / 42.88025 µs / 29.48086 µs / 0.0 / 0.00163 / 0.00326 / 0.02925
        Q3 / 5.43494 GHz / 46.88035 µs / 17.74449 µs / 0.0 / 0.00189 / 0.00378 / 0.023
        Q4 / 5.17582 GHz / 41.22472 µs / 11.09655 µs / 0.0 / 0.00335 / 0.0067 / 0.07525
    
    Multi-Qubit Gates [Name / Type / Gate Error]
    --------------------------------------------
        CX1_0 / cx / 0.03639
        CX2_0 / cx / 0.02608
        CX2_1 / cx / 0.04075
        CX3_2 / cx / 0.06022
        CX3_4 / cx / 0.04131
        CX4_2 / cx / 0.06126


Or, if we are interested in a higher-level view of all the backends
available to us, then we can use ``backend_overview()``

.. code:: ipython3

    backend_overview()


.. parsed-literal::

    ibmq_20_tokyo               ibmq_16_melbourne            ibmqx4
    -------------               -----------------            ------
    Num. Qubits:  20            Num. Qubits:  14             Num. Qubits:  5
    Pending Jobs: 0             Pending Jobs: 3              Pending Jobs: 6
    Least busy:   True          Least busy:   False          Least busy:   False
    Operational:  True          Operational:  True           Operational:  True
    Avg. T1:      86.9          Avg. T1:      50.3           Avg. T1:      47.2
    Avg. T2:      55.3          Avg. T2:      63.0           Avg. T2:      24.7
    
    
    


There are also Jupyter magic equivalents that give more detailed
information.

.. code:: ipython3

    %qiskit_backend_monitor backend


.. parsed-literal::

    VBox(children=(HTML(value="<h1 style='color:#ffffff;background-color:#000000;padding-top: 1%;padding-bottom: 1…


The Jupyter ``backend_overview`` runs live in the notebook, and will
automatically update itself every minute.

.. code:: ipython3

    %qiskit_backend_overview



.. parsed-literal::

    VBox(children=(HTML(value="<h2 style ='color:#ffffff; background-color:#000000;padding-top: 1%; padding-bottom…

