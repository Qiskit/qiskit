#####################################################
Compute circuit output probabilities with ``Sampler``
#####################################################

This guide shows how to get the probability distribution of a quantum circuit with the :class:`~qiskit.primitives.Sampler` primitive.

.. note::

    While this guide only uses Qiskit Terra's implementation of the ``Sampler`` primitive, there are other
    implementations of this primitive like Qiskit Terra's :class:`~qiskit.primitives.BackendSampler`, Qiskit Aer's :class:`~qiskit_aer.primitives.Sampler`
    and Qiskit Runtime's :class:`~qiskit_ibm_runtime.Sampler`.

Initialize quantum circuits
===========================

The first step is to create the :class:`~qiskit.circuit.QuantumCircuit` from which you want to obtain the probability distribution.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()
    qc.draw("mpl")

.. testsetup::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()

Initialize the ``Sampler``
==========================

Then, you need to create a :class:`~qiskit.primitives.Sampler` object.

.. testcode::

    from qiskit.primitives import Sampler

    sampler = Sampler()

Run and get results
===================

Now that you have defined ``sampler``, you can create a :class:`~.PrimitiveJob` (subclass of :class:`~qiskit.providers.JobV1`) with the
:meth:`~qiskit.primitives.Sampler.run` method and, then, you can get the results (as a :class:`~qiskit.primitives.SamplerResult` object) with
the results with the :meth:`~qiskit.providers.JobV1.result` method.

.. testcode::

    job = sampler.run(qc)
    result = job.result()
    print(result)

.. testoutput::

    SamplerResult(quasi_dists=[{0: 0.4999999999999999, 3: 0.4999999999999999}], metadata=[{}])

Get the probability distribution
--------------------------------

From these results you can take the probability distributions with the attribute :attr:`~qiskit.primitives.SamplerResult.quasi_dists`.

Even though there is only one circuit in this example, :attr:`~qiskit.primitives.SamplerResult.quasi_dists` returns a list of :class:`~qiskit.result.QuasiDistribution`\ s.
Generally ``result.quasi_dists[i]`` would be the quasi-probability distribution of the ``i``-th circuit.

.. testcode::

    quasi_dist = result.quasi_dists[0]
    print(quasi_dist)


.. testoutput::

    {0: 0.4999999999999999, 3: 0.4999999999999999}

Probability distribution with binary outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to see the outputs as binary strings instead of decimal, you can use the
:meth:`~qiskit.result.QuasiDistribution.binary_probabilities` method.

.. testcode::
    
    print(quasi_dist.binary_probabilities())

.. testoutput::

    {'00': 0.4999999999999999, '11': 0.4999999999999999}

Parameterized circuits with ``Sampler``
=========================================

The :class:`~qiskit.primitives.Sampler` primitive also has the option to include unbound parameterized circuits like the one below.
You can also bind values to the parameters of the circuit and follow the steps
of the previous example.

.. testcode::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0,1)
    qc.measure_all()
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

            ┌───────┐      ░ ┌─┐   
       q_0: ┤ Ry(θ) ├──■───░─┤M├───
            └───────┘┌─┴─┐ ░ └╥┘┌─┐
       q_1: ─────────┤ X ├─░──╫─┤M├
                     └───┘ ░  ║ └╥┘
    meas: 2/══════════════════╩══╩═
                              0  1 

The main difference from the previous case is that now you need to include the parameter values
for which you want to evaluate the expectation value as a ``list`` of ``list``\ s of ``float``\ s.
The idea is that the ``i``-th element of the bigger ``list`` is the set of parameter values
that corresponds to the ``i``-th circuit.

.. testcode::

    import numpy as np

    parameter_values = [[0], [np.pi/6], [np.pi/2]]

    job = sampler.run([qc]*3, parameter_values=parameter_values)
    dists = job.result().quasi_dists

    for i in range(3):
        print(f"Parameter: {parameter_values[i][0]:.5f}\t Probabilities: {dists[i]}")

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Parameter: 0.00000	 Probabilities: {0: 1.0}
    Parameter: 0.52360	 Probabilities: {0: 0.9330127018922194, 3: 0.0669872981077807}
    Parameter: 1.57080	 Probabilities: {0: 0.5000000000000001, 3: 0.4999999999999999}
