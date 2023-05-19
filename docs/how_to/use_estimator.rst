#########################################################
Compute an expectation value with ``Estimator`` primitive
#########################################################

This guide shows how to get the expected value of an observable for a given quantum circuit with the :class:`~qiskit.primitives.Estimator` primitive.

.. note::

    While this guide uses Qiskit’s reference implementation, the ``Estimator`` primitive can be run with any provider using :class:`~qiskit.primitives.BackendEstimator` .
    
    .. code-block::

        from qiskit.primitives import BackendEstimator
        from <some_qiskit_provider> import QiskitProvider

        provider = QiskitProvider()
        backend = provider.get_backend('backend_name')
        estimator = BackendEstimator(backend)

    There are some providers that implement primitives natively (see `this page <http://qiskit.org/providers/#primitives>`_ for more details).


Initialize observables
======================

The first step is to define the observables whose expected value you want to compute. Each observable can be any ``BaseOperator``, like the operators from :mod:`qiskit.quantum_info`.
Among them it is preferable to use :class:`~qiskit.quantum_info.SparsePauliOp`.

.. testcode::

    from qiskit.quantum_info import SparsePauliOp

    observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])

Initialize quantum circuit
==========================

Then you need to create the :class:`~qiskit.circuit.QuantumCircuit`\ s for which you want to obtain the expected value.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.draw("mpl")

.. testsetup::

    # This code is repeated (but hidden) because we will need to use the variables with the extension sphinx.ext.doctest (testsetup/testcode/testoutput directives)
    # and we can't reuse the variables from the plot directive above because they are incompatible.
    # The plot directive is used to draw the circuit with matplotlib and the code is shown because of the include-source flag.
    
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)

.. note::

    The :class:`~qiskit.circuit.QuantumCircuit` you pass to :class:`~qiskit.primitives.Estimator` must not include any measurements.

Initialize the ``Estimator``
============================

Then, you need to instantiate an :class:`~qiskit.primitives.Estimator`.

.. testcode::

    from qiskit.primitives import Estimator

    estimator = Estimator()

Run and get results
===================

Now that you have defined your ``estimator``, you can run your estimation by calling the :meth:`~qiskit.primitives.Estimator.run` method, 
which returns an instance of :class:`~.PrimitiveJob` (subclass of :class:`~qiskit.providers.JobV1`). You can get the results from the job (as a :class:`~qiskit.primitives.EstimatorResult` object) 
with the :meth:`~qiskit.providers.JobV1.result` method.

.. testcode::

    job = estimator.run(qc, observable)
    result = job.result()
    print(result)

.. testoutput::

    EstimatorResult(values=array([4.]), metadata=[{}])

While this example only uses one :class:`~qiskit.circuit.QuantumCircuit` and one observable, if you want to get expectation values for multiple circuits and observables you can
pass a ``list`` of :class:`~qiskit.circuit.QuantumCircuit`\ s and a list of ``BaseOperator``\ s to the :meth:`~qiskit.primitives.Estimator.run` method. Both ``list``\ s must have
the same length.

Get the expected value
----------------------

From these results you can extract the expected values with the attribute :attr:`~qiskit.primitives.EstimatorResult.values`.

:attr:`~qiskit.primitives.EstimatorResult.values` returns a :class:`numpy.ndarray`
whose ``i``-th element is the expectation value corresponding to the ``i``-th circuit and ``i``-th observable.

.. testcode::

    exp_value = result.values[0]
    print(exp_value)

.. testoutput::

    3.999999999999999

Parameterized circuit with ``Estimator``
========================================

The :class:`~qiskit.primitives.Estimator` primitive can be run with unbound parameterized circuits like the one below.
You can also manually bind values to the parameters of the circuit and follow the steps
of the previous example.

.. testcode::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')
    param_qc = QuantumCircuit(2)
    param_qc.ry(theta, 0)
    param_qc.cx(0,1)
    print(param_qc.draw())

.. testoutput::

         ┌───────┐     
    q_0: ┤ Ry(θ) ├──■──
         └───────┘┌─┴─┐
    q_1: ─────────┤ X ├
                  └───┘

The main difference with the previous case is that now you need to specify the sets of parameter values
for which you want to evaluate the expectation value as a ``list`` of ``list``\ s of ``float``\ s.
The ``i``-th element of the outer``list`` is the set of parameter values
that corresponds to the ``i``-th circuit and observable.

.. testcode::

    import numpy as np
    
    parameter_values = [[0], [np.pi/6], [np.pi/2]]

    job = estimator.run([param_qc]*3, [observable]*3, parameter_values=parameter_values)
    values = job.result().values

    for i in range(3):
        print(f"Parameter: {parameter_values[i][0]:.5f}\t Expectation value: {values[i]}")

.. testoutput::

    Parameter: 0.00000	 Expectation value: 2.0
    Parameter: 0.52360	 Expectation value: 3.0
    Parameter: 1.57080	 Expectation value: 4.0

Change run options
==================

Your workflow might require tuning primitive run options, such as the amount of shots.

By default, the reference :class:`~qiskit.primitives.Estimator` class performs an exact statevector
calculation based on the :class:`~qiskit.quantum_info.Statevector` class. However, this can be 
modified to include shot noise if the number of ``shots`` is set. 
For reproducibility purposes, a ``seed`` will also be set in the following examples.

There are two main ways of setting options in the :class:`~qiskit.primitives.Estimator`:

* Set keyword arguments in the :meth:`~qiskit.primitives.Estimator.run` method.
* Modify :class:`~qiskit.primitives.Estimator` options.

Set keyword arguments for :meth:`~qiskit.primitives.Estimator.run`
------------------------------------------------------------------

If you only want to change the settings for a specific run, it can be more convenient to
set the options inside the :meth:`~qiskit.primitives.Estimator.run` method. You can do this by
passing them as keyword arguments.

.. testcode::

    job = estimator.run(qc, observable, shots=2048, seed=123)
    result = job.result()
    print(result)

.. testoutput::

    EstimatorResult(values=array([4.]), metadata=[{'variance': 3.552713678800501e-15, 'shots': 2048}])

.. testcode::

    print(result.values[0])

.. testoutput::

    3.999999998697238

Modify :class:`~qiskit.primitives.Estimator` options
-----------------------------------------------------

If you want to keep some configuration values for several runs, it can be better to
change the :class:`~qiskit.primitives.Estimator` options. That way you can use the same 
:class:`~qiskit.primitives.Estimator` object as many times as you wish without having to
rewrite the configuration values every time you use :meth:`~qiskit.primitives.Estimator.run`.

Modify existing :class:`~qiskit.primitives.Estimator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to change the options of an already-defined :class:`~qiskit.primitives.Estimator`, you can use
:meth:`~qiskit.primitives.Estimator.set_options` and introduce the new options as keyword arguments.

.. testcode::

    estimator.set_options(shots=2048, seed=123)

    job = estimator.run(qc, observable)
    result = job.result()
    print(result)

.. testoutput::

    EstimatorResult(values=array([4.]), metadata=[{'variance': 3.552713678800501e-15, 'shots': 2048}])

.. testcode::

    print(result.values[0])

.. testoutput::

    3.999999998697238


Define a new :class:`~qiskit.primitives.Estimator` with the options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to define a new :class:`~qiskit.primitives.Estimator` with new options, you need to
define a ``dict`` like this one:

.. testcode::

    options = {"shots": 2048, "seed": 123}

And then you can introduce it into your new :class:`~qiskit.primitives.Estimator` with the
``options`` argument.

.. testcode::

    estimator = Estimator(options=options)

    job = estimator.run(qc, observable)
    result = job.result()
    print(result)

.. testoutput::

    EstimatorResult(values=array([4.]), metadata=[{'variance': 3.552713678800501e-15, 'shots': 2048}])

.. testcode::

    print(result.values[0])

.. testoutput::

    3.999999998697238