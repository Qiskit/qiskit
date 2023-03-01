#########################
Use the Sampler primitive
#########################

This guide shows how to get the probability distribution of a quantum circuit with the :class:`~qiskit.primitives.Sampler` primitive.

Initialize quantum circuits
===========================

The first step is to create the :class:`~qiskit.circuit.QuantumCircuit` from which you want to obtain the probability distribution.
For more details about this part check out :doc:`this guide <create_a_quantum_circuit>`.

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

Even though there is only one circuit in this example, :attr:`~qiskit.primitives.SamplerResult.quasi_dists` returns a list of :class:`~qiskit.result.QuasiDistribution` s.
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
