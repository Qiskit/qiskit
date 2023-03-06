###########################
Use the Estimator primitive
###########################

This guide shows how to get the expected value of an observable for a given quantum circuit with the :class:`~qiskit.primitives.Estimator` primitive.

.. note::

    While this guide only uses Qiskit Terra's implementation of the ``Estimator`` primitive, there are other
    implementations of this primitive like Qiskit Terra's :class:`~qiskit.primitives.BackendEstimator`, Qiskit Aer's :class:`~qiskit_aer.primitives.Estimator`
    and Qiskit Runtime's :class:`~qiskit_ibm_runtime.Estimator`.


Initialize observable
=====================

The first step is to define the observable whose expected value you want to compute. This observable can be any ``BaseOperator``, like the operators from :mod:`qiskit.quantum_info`.
Among them it is preferable to use :class:`~qiskit.quantum_info.SparsePauliOp`.

.. testcode::

    from qiskit.quantum_info import SparsePauliOp

    obs = SparsePauliOp(["II", "XX", "YY", "ZZ"], [1, 1, -1, 1])

Initialize quantum circuits
===========================

Then you need to create the :class:`~qiskit.circuit.QuantumCircuit` for which you want to obtain the expected value.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.draw("mpl")

.. testsetup::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)

Initialize the ``Estimator``
============================

Then, you need to create an :class:`~qiskit.primitives.Estimator` object.

.. testcode::

    from qiskit.primitives import Estimator

    estimator = Estimator()

Run and get results
===================

Now that you have defined ``estimator``, you can create a :class:`~.PrimitiveJob` (subclass of :class:`~qiskit.providers.JobV1`) with the
:meth:`~qiskit.primitives.Estimator.run` method and, then, you can get the results (as a :class:`~qiskit.primitives.EstimatorResult` object) with
the results with the :meth:`~qiskit.providers.JobV1.result` method.

.. testcode::

    job = estimator.run(qc, obs)
    result = job.result()
    print(result)

.. testoutput::

    EstimatorResult(values=array([4.]), metadata=[{}])

Get the expected value
----------------------

From these results you can take the expected values with the attribute :attr:`~qiskit.primitives.EstimatorResult.values`.

Generally, :attr:`~qiskit.primitives.EstimatorResult.values` returns a `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
whose ``i``-th element would be the expectation value corresponding to the ``i``-th circuit and ``i``-th observable.

.. testcode::

    exp_value = result.values[0]
    print(exp_value)


.. testoutput::

    3.999999999999999

Parameterized circuits with ``Estimator``
=========================================

The :class:`~qiskit.primitives.Estimator` primitive also has the option to include unbound parameterized circuits like the one below.
You can also bind values to the parameters of the circuit and follow the steps
of the previous example.

.. testcode::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0,1)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───────┐     
    q_0: ┤ Ry(θ) ├──■──
         └───────┘┌─┴─┐
    q_1: ─────────┤ X ├
                  └───┘

The main difference from the previous case is that now you need to include the parameter values
for which you want to evaluate the expectation value as a ``list`` of ``list``\ s of ``float``\ s.
The idea is that the ``i``-th element of the bigger ``list`` is the set of parameter values
that corresponds to the ``i``-th circuit and observable.

.. testcode::

    import numpy as np
    
    parameter_values = [[0], [np.pi/6], [np.pi/2]]

    job = estimator.run([qc]*3, [obs]*3, parameter_values=parameter_values)
    values = job.result().values

    for i in range(3):
        print(f"Parameter: {parameter_values[i][0]:.5f}\t Expectation value: {values[i]}")

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Parameter: 0.00000	 Expectation value: 2.0
    Parameter: 0.52360	 Expectation value: 3.0
    Parameter: 1.57080	 Expectation value: 4.0


