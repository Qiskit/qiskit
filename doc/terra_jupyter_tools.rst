


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

    /Users/jaygambetta/gitshare/qiskit-terra/qiskit/providers/ibmq/ibmqbackend.py:180: DeprecationWarning: Some jobs (281) are in a no-longer supported format. These jobs will stop working after Qiskit 0.7. Save the results or send the job with Qiskit 0.7+. Old jobs:
     - 5c156b801d475200611467ee
     - 5c156b2a737f7600571ec66a
     - 5c156a49557dff00576be243
     - 5c15694eee03b40054b35f45
     - 5c156921f07ad5005c5df729
     - 5c156877a57edb00531278e1
     - 5c156851ed804c00561942b7
     - 5c15682513eb0c005321c326
     - 5c156762d38acd00557c9bda
     - 5c15672a737f7600571ec65e
     - 5c1565fded804c00561942ac
     - 5c1564ccaa0f280056346f61
     - 5c15559ba57edb00531278ae
     - 5c0ac161751a790052faca8e
     - 5c0ac01d1d0f11005de5d7c1
     - 5c0abee6ab6cf6005c8d312c
     - 5bfe0ec5c9630c0055d26435
     - 5bfe0e0f3626d5005e226360
     - 5be8f55b28716d00547b70ad
     - 5be8f54e9a9893006ff69200
     - 5be8f39c54dacb0059c2b0db
     - 5be8f394afd471005540d4d7
     - 5be8ae5e17436b0052751909
     - 5be748a7e00f60005ad7f23d
     - 5be746e3d4d36f0054595d60
     - 5bc3e88d404ceb006174af14
     - 5bc3e84b21da3300548def31
     - 5bc3e2f6860686005e02231e
     - 5bc3e1f621da3300548def27
     - 5bc3dff65b5a470061ad9e40
     - 5bc3dfe14a2574005fdb3c45
     - 5bc3df65c2c7f6005ff968e8
     - 5bc3df3821da3300548def23
     - 5bc3ded8249a3c006799e341
     - 5bc3de9f942db700650521f6
     - 5bc2a4315b5a470061ad9aa0
     - 5bc29dd921da3300548deafe
     - 5bc29d8bee99500060e0dfdb
     - 5bc29d5960af9300675db78b
     - 5bb15f0d2cc5e80038795517
     - 5bb15ebccb2b78003912b463
     - 5bb15de947fe5e0042fb2e0f
     - 5bb15dcb6744c7003942d56d
     - 5bb15d7d2cc5e80038795510
     - 5bb159562878c20039998f27
     - 5bb1542a5fa98d003c0a9428
     - 5bb14d0e7759d400407f5c66
     - 5bb14c357759d400407f5c63
     - 5bb1462d2cc5e80038795491
     - 5bb1460c07fb5400519d1c68
     - 5ba567ae90285c0037809333
     - 5ba5605a8a925c003f901d38
     - 5ba55a070d743f0050ffee9e
     - 5ba559bdfa913100395ac396
     - 5ba557d220996b003bf17b30
     - 5ba551fcc6106f003cb4fd59
     - 5ba54ea523cfc7003c64a0f2
     - 5ba54e51fa913100395ac338
     - 5ba54c59e71ae70042e0117c
     - 5ba309cf7a6a570039fc1547
     - 5ba3083fe19f200039abfc4f
     - 5b988abb3a92f6003a02ec78
     - 5b6e22501d4b000045d04a0b
     - 5b6e2087f898a1003ae6d683
     - 5b6e20721d4b000045d04a02
     - 5b6e1ff1f898a1003ae6d681
     - 5b6e1f8cb127f500382924c0
     - 5b68a2b3358efc0039e47b89
     - 5b686a35a35320004b5cdf6c
     - 5b68634b05757400389ea560
     - 5b686059b2e101003a4ba943
     - 5b686031f9b81b00364b2230
     - 5b685acef9b81b00364b221f
     - 5b683987113f850038b76c50
     - 5b683979e09a8f003b0ef580
     - 5b68393ea35320004b5cde66
     - 5b67dfcd8fb0a1003d8a3f92
     - 5b67df7bd3234c0038778adf
     - 5b67de53d26f0f0039cf90fa
     - 5b67d93c1d4b000045d03149
     - 5b67c2311d4b000045d03114
     - 5b67c1de4217e60038b3f34d
     - 5b67bdf7f79b7200366a2603
     - 5b67bd6ba35320004b5cdd2c
     - 5b5f59fda98e23003a622225
     - 5b50dcd6b221ab0039ab6188
     - 5b50d2160e4c1c00404c89a6
     - 5b50c27bcc3090004e016f93
     - 5b50a09f3382e6003c7d0f80
     - 5b4fa6e238fe5e003e1d2303
     - 5b4f7f2ba90c3f003f7e40d8
     - 5b4eacaf58e3fb00384b4c84
     - 5b4eac59b221ab0039ab5185
     - 5b4eac2211322d003bf0e806
     - 5b4eab9f58e3fb00384b4c80
     - 5b4ea163fc5378003bf3cf3a
     - 5b4e8f9a177df5003bf44c57
     - 5b492083f5c60500377a1158
     - 5b491fab58e3fb00384b3279
     - 5b491f69177df5003bf43169
     - 5b491f5038fe5e003e1d02b2
     - 5b49195788ab3700472094d0
     - 5b4918cc88ab37004720949d
     - 5b4916821deae90038493c64
     - 5b49167b58e3fb00384b3223
     - 5b491669e036a500391bd7cb
     - 5b4916520e4c1c00404c600e
     - 5b4915fd58e3fb00384b321d
     - 5b4915d211322d003bf0cc0f
     - 5b17dbac3092d6003cacef7d
     - 5b17bdd7e8beb60042becebb
     - 5b1750742c62ef003ba379a2
     - 5af9a01594ac68003debd730
     - 5af9a04594ac68003debea52
     - 5af9a01494ac68003debd6eb
     - 5af9a04294ac68003debe917
     - 5af9a03194ac68003debe4cc
     - 5af9a04594ac68003debea57
     - 5af9a04294ac68003debe8f4
     - 5af9a02b94ac68003debe1d5
     - 5af9a04d94ac68003debec09
     - 5af9a02a94ac68003debe11b
     - 5af9a00694ac68003debd24a
     - 5af9a01c94ac68003debda43
     - 5af9a01094ac68003debd5ab
     - 5af9a03294ac68003debe5a2
     - 5af99f9e94ac68003debba1f
     - 5af9a03194ac68003debe4c6
     - 5af9a03194ac68003debe4de
     - 5af9a02e94ac68003debe389
     - 5af9a01b94ac68003debd99b
     - 5af99ffb94ac68003debcede
     - 5af9a02594ac68003debdecd
     - 5af9a02e94ac68003debe339
     - 5af9a01b94ac68003debd97d
     - 5af9a03094ac68003debe436
     - 5af9a03294ac68003debe5c7
     - 5af9a02394ac68003debde07
     - 5af99fca94ac68003debbf3b
     - 5af99fb494ac68003debbbcf
     - 5af9a05a94ac68003debf056
     - 5af9a04294ac68003debe8fc
     - 5af9a05b94ac68003debf05c
     - 5af99fdf94ac68003debc72b
     - 5af9a03594ac68003debe69c
     - 5af9a05794ac68003debeee3
     - 5af9a02294ac68003debdd11
     - 5af9a02294ac68003debdd32
     - 5af9a02294ac68003debdd36
     - 5af9a01f94ac68003debdb9a
     - 5af99fd394ac68003debc1a6
     - 5af9a01e94ac68003debdb19
     - 5af99fba94ac68003debbc6f
     - 5af99fdc94ac68003debc5ab
     - 5af99fc294ac68003debbd3e
     - 5af99fe894ac68003debc982
     - 5ae91de40f020500399c7c94
     - 5ae91cec0f020500399c374a
     - 5ae915240f020500399788fb
     - 5ae91d010f020500399c3dac
     - 5ae91dbe0f020500399c7446
     - 5ae91b250f020500399b3dfa
     - 5ae9151e0f0205003997838e
     - 5ae91cdb0f020500399c2f7e
     - 5ae9151e0f0205003997838f
     - 5ae91dee0f020500399c7d8a
     - 5ae91e3a0f020500399c9046
     - 5ae91e3a0f020500399c9056
     - 5ae915060f02050039977add
     - 5ae91d010f020500399c3da9
     - 5ae91dbe0f020500399c743d
     - 5ae91e320f020500399c8d0a
     - 5ae91db50f020500399c729a
     - 5ae91db50f020500399c729c
     - 5ae91da70f020500399c6e7e
     - 5ae91dee0f020500399c7d78
     - 5ae91af20f020500399b1d40
     - 5ae91e3a0f020500399c9040
     - 5ae91dcc0f020500399c78db
     - 5ae91e120f020500399c861a
     - 5ae91dc30f020500399c75f7
     - 5ae91d440f020500399c59a9
     - 5ae91bf70f020500399ba176
     - 5ae91bdd0f020500399b923f
     - 5ae91bde0f020500399b925a
     - 5ae91ce30f020500399c3370
     - 5ae91a790f020500399ae047
     - 5ae914160f0205003996cf02
     - 5ae91c390f020500399bc7b4
     - 5ae91ce00f020500399c31bd
     - 5ae914160f0205003996cef9
     - 5ae91c250f020500399bb80d
     - 5ae91c9f0f020500399c0ea4
     - 5ae91c2e0f020500399bbef7
     - 5ae91c620f020500399be48e
     - 5ae916c30f02050039987e50
     - 5ae91a150f020500399aa37d
     - 5ae91bba0f020500399b83ae
     - 5ae91ada0f020500399b0cda
     - 5ae91afb0f020500399b2386
     - 5ae91d330f020500399c5135
     - 5ae91be00f020500399b93cf
     - 5ae916710f02050039983c93
     - 5ae91d410f020500399c58cb
     - 5ae91d230f020500399c4624
     - 5ae91d200f020500399c44a7
     - 5ae91b080f020500399b2aa8
     - 5ae91bb00f020500399b7ca1
     - 5ae9166d0f0205003998398c
     - 5ae914580f020500399705b5
     - 5ae91bb00f020500399b7c98
     - 5ae91cb80f020500399c1d70
     - 5ae91aea0f020500399b1894
     - 5ae91bb00f020500399b7ca8
     - 5ae91a150f020500399aa38f
     - 5ae9142b0f0205003996de91
     - 5ae918260f020500399971fa
     - 5ae919690f020500399a474c
     - 5ae902de0f020500398de1cf
     - 5ae917900f0205003999138e
     - 5ae918ff0f020500399a04e9
     - 5ae9184e0f02050039999086
     - 5ae919010f020500399a059d
     - 5ae918f00f0205003999fb8e
     - 5ae9182e0f02050039997838
     - 5ae90eaa0f0205003993ad6d
     - 5ae917ea0f02050039995384
     - 5ae919220f020500399a16d9
     - 5ae918430f02050039998981
     - 5ae919010f020500399a05b4
     - 5ae917d80f020500399947d2
     - 5ae90e490f02050039936a53
     - 5ae913220f020500399639c4
     - 5ae912ff0f02050039962a2e
     - 5ae9151a0f02050039978125
     - 5ae919370f020500399a24fe
     - 5ae919380f020500399a25c4
     - 5ae919420f020500399a2ccb
     - 5ae916a40f0205003998648b
     - 5ae9156e0f0205003997b62c
     - 5ae9168d0f020500399850db
     - 5ae9156e0f0205003997b62e
     - 5ae917ae0f02050039992e13
     - 5ae911940f02050039955bfa
     - 5ae911e00f02050039958dc5
     - 5ae911df0f02050039958d37
     - 5ae911e00f02050039958dc8
     - 5ae917970f02050039991a90
     - 5ae9189a0f0205003999ccb4
     - 5ae914720f02050039971940
     - 5ae9189a0f0205003999ccaa
     - 5ae910cf0f0205003994e926
     - 5ae9146b0f020500399713bd
     - 5ae90c150f02050039928d31
     - 5ae90c120f02050039928b98
     - 5ae90f490f020500399413ba
     - 5ae912840f0205003995e963
     - 5ae90f490f020500399413a9
     - 5ae918a30f0205003999d3bb
     - 5ae910990f0205003994bfb8
     - 5ae90f560f02050039941ec5
     - 5ae916c60f02050039988167
     - 5ae910430f0205003994957a
     - 5ae913900f020500399680f6
     - 5ae910160f0205003994796b
     - 5ae918730f0205003999afc5
     - 5ae918710f0205003999ae38
     - 5ae9104b0f02050039949ac1
     - 5ae911720f0205003995422c
     - 5ae9183d0f020500399984d2
     - 5ae910080f0205003994722c
     - 5ae913960f020500399684d2
     - 5ae910080f0205003994723c
     - 5ae91c0f0f020500399bac0a
     - 5ae90b220f02050039921f02
     - 5ae916020f0205003997e785
     - 5ae913950f02050039968411
     - 5ae90b9e0f0205003992510e
     - 5ae90b220f02050039921efb
     - 5ae90eab0f0205003993ae33
     - 5ae90b5f0f02050039923e28
      DeprecationWarning)



.. parsed-literal::

    VBox(children=(HTML(value="<h1 style='color:#ffffff;background-color:#000000;padding-top: 1%;padding-bottom: 1…


The Jupyter ``backend_overview`` runs live in the notebook, and will
automatically update itself every minute.

.. code:: ipython3

    %qiskit_backend_overview



.. parsed-literal::

    VBox(children=(HTML(value="<h2 style ='color:#ffffff; background-color:#000000;padding-top: 1%; padding-bottom…

