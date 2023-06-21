###############################################################
Compute circuit output probabilities with ``Sampler`` primitive
###############################################################

This guide shows how to get the probability distribution of a quantum circuit with the :class:`~qiskit.primitives.Sampler` primitive.

.. note::

    While this guide uses Qiskit’s reference implementation, the ``Sampler`` primitive can be run with any provider using :class:`~qiskit.primitives.BackendSampler`.
    
    .. code-block::

        from qiskit.primitives import BackendSampler
        from <some_qiskit_provider> import QiskitProvider

        provider = QiskitProvider()
        backend = provider.get_backend('backend_name')
        sampler = BackendSampler(backend)

    There are some providers that implement primitives natively (see `this page <http://qiskit.org/providers/#primitives>`_ for more details).

Initialize quantum circuits
===========================

The first step is to create the :class:`~qiskit.circuit.QuantumCircuit`\ s from which you want to obtain the probability distribution.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()
    qc.draw("mpl")

.. testsetup::

    # This code is repeated (but hidden) because we will need to use the variables with the extension sphinx.ext.doctest (testsetup/testcode/testoutput directives)
    # and we can't reuse the variables from the plot directive above because they are incompatible.
    # The plot directive is used to draw the circuit with matplotlib and the code is shown because of the include-source flag.

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()

.. note::

    The :class:`~qiskit.circuit.QuantumCircuit` you pass to :class:`~qiskit.primitives.Sampler` has to include measurements.

Initialize the ``Sampler``
==========================

Then, you need to create a :class:`~qiskit.primitives.Sampler` instance.

.. testcode::

    from qiskit.primitives import Sampler

    sampler = Sampler()

Run and get results
===================

Now that you have defined your ``sampler``, you can run it by calling the :meth:`~qiskit.primitives.Sampler.run` method, 
which returns an instance of :class:`~.PrimitiveJob` (subclass of :class:`~qiskit.providers.JobV1`). You can get the results from the job (as a :class:`~qiskit.primitives.SamplerResult` object) 
with the :meth:`~qiskit.providers.JobV1.result` method.

.. testcode::

    job = sampler.run(qc)
    result = job.result()
    print(result)

.. testoutput::

    SamplerResult(quasi_dists=[{0: 0.4999999999999999, 3: 0.4999999999999999}], metadata=[{}])

While this example only uses one :class:`~qiskit.circuit.QuantumCircuit`, if you want to sample multiple circuits you can
pass a ``list`` of :class:`~qiskit.circuit.QuantumCircuit` instances to the :meth:`~qiskit.primitives.Sampler.run` method.

Get the probability distribution
--------------------------------

From these results you can extract the quasi-probability distributions with the attribute :attr:`~qiskit.primitives.SamplerResult.quasi_dists`.

Even though there is only one circuit in this example, :attr:`~qiskit.primitives.SamplerResult.quasi_dists` returns a list of :class:`~qiskit.result.QuasiDistribution`\ s.
``result.quasi_dists[i]`` is the quasi-probability distribution of the ``i``-th circuit.

.. note::

    A quasi-probability distribution differs from a probability distribution in that negative values are also allowed.
    However the quasi-probabilities must sum up to 1 like probabilities.
    Negative quasi-probabilities may appear when using error mitigation techniques.

.. testcode::

    quasi_dist = result.quasi_dists[0]
    print(quasi_dist)

.. testoutput::

    {0: 0.4999999999999999, 3: 0.4999999999999999}

Probability distribution with binary outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to see the output keys as binary strings instead of decimal numbers, you can use the
:meth:`~qiskit.result.QuasiDistribution.binary_probabilities` method.

.. testcode::
    
    print(quasi_dist.binary_probabilities())

.. testoutput::

    {'00': 0.4999999999999999, '11': 0.4999999999999999}

Parameterized circuit with ``Sampler``
========================================

The :class:`~qiskit.primitives.Sampler` primitive can be run with unbound parameterized circuits like the one below.
You can also manually bind values to the parameters of the circuit and follow the steps
of the previous example.

.. testcode::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')
    param_qc = QuantumCircuit(2)
    param_qc.ry(theta, 0)
    param_qc.cx(0,1)
    param_qc.measure_all()
    print(param_qc.draw())

.. testoutput::

            ┌───────┐      ░ ┌─┐   
       q_0: ┤ Ry(θ) ├──■───░─┤M├───
            └───────┘┌─┴─┐ ░ └╥┘┌─┐
       q_1: ─────────┤ X ├─░──╫─┤M├
                     └───┘ ░  ║ └╥┘
    meas: 2/══════════════════╩══╩═
                              0  1 

The main difference from the previous case is that now you need to specify the sets of parameter values
for which you want to evaluate the expectation value as a ``list`` of ``list``\ s of ``float``\ s.
The ``i``-th element of the outer ``list`` is the set of parameter values
that corresponds to the ``i``-th circuit.

.. testcode::

    import numpy as np

    parameter_values = [[0], [np.pi/6], [np.pi/2]]

    job = sampler.run([param_qc]*3, parameter_values=parameter_values)
    dists = job.result().quasi_dists

    for i in range(3):
        print(f"Parameter: {parameter_values[i][0]:.5f}\t Probabilities: {dists[i]}")

.. testoutput::

    Parameter: 0.00000	 Probabilities: {0: 1.0}
    Parameter: 0.52360	 Probabilities: {0: 0.9330127018922194, 3: 0.0669872981077807}
    Parameter: 1.57080	 Probabilities: {0: 0.5000000000000001, 3: 0.4999999999999999}

Change run options
==================

Your workflow might require tuning primitive run options, such as the amount of shots.

By default, the reference :class:`~qiskit.primitives.Sampler` class performs an exact statevector
calculation based on the :class:`~qiskit.quantum_info.Statevector` class. However, this can be 
modified to include shot noise if the number of ``shots`` is set. 
For reproducibility purposes, a ``seed`` will also be set in the following examples.

There are two main ways of setting options in the :class:`~qiskit.primitives.Sampler`:

* Set keyword arguments in the :meth:`~qiskit.primitives.Sampler.run` method.
* Modify :class:`~qiskit.primitives.Sampler` options.

Set keyword arguments for :meth:`~qiskit.primitives.Sampler.run`
----------------------------------------------------------------

If you only want to change the settings for a specific run, it can be more convenient to
set the options inside the :meth:`~qiskit.primitives.Sampler.run` method. You can do this by
passing them as keyword arguments.

.. testcode::

    job = sampler.run(qc, shots=2048, seed=123)
    result = job.result()
    print(result)

.. testoutput::

    SamplerResult(quasi_dists=[{0: 0.5205078125, 3: 0.4794921875}], metadata=[{'shots': 2048}])

Modify :class:`~qiskit.primitives.Sampler` options
---------------------------------------------------

If you want to keep some configuration values for several runs, it can be better to
change the :class:`~qiskit.primitives.Sampler` options. That way you can use the same 
:class:`~qiskit.primitives.Sampler` object as many times as you wish without having to
rewrite the configuration values every time you use :meth:`~qiskit.primitives.Sampler.run`.

Modify existing :class:`~qiskit.primitives.Sampler`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to change the options of an already-defined :class:`~qiskit.primitives.Sampler`, you can use
:meth:`~qiskit.primitives.Sampler.set_options` and introduce the new options as keyword arguments.

.. testcode::

    sampler.set_options(shots=2048, seed=123)

    job = sampler.run(qc)
    result = job.result()
    print(result)

.. testoutput::

    SamplerResult(quasi_dists=[{0: 0.5205078125, 3: 0.4794921875}], metadata=[{'shots': 2048}])

Define a new :class:`~qiskit.primitives.Sampler` with the options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to define a new :class:`~qiskit.primitives.Sampler` with new options, you need to
define a ``dict`` like this one:

.. testcode::

    options = {"shots": 2048, "seed": 123}

And then you can introduce it into your new :class:`~qiskit.primitives.Sampler` with the
``options`` argument.

.. testcode::

    sampler = Sampler(options=options)

    job = sampler.run(qc)
    result = job.result()
    print(result)

.. testoutput::

    SamplerResult(quasi_dists=[{0: 0.5205078125, 3: 0.4794921875}], metadata=[{'shots': 2048}])