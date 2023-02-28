##############################
Create a parameterized circuit
##############################

This guide will show how to create a :class:`~.QuantumCircuit` that includes parameters.

Define the parameters
=====================

In order to define a parameter, you need to create a :class:`~.Parameter` object. To do that you only need to choose a ``name``, that can be any unicode string like ``'θ'``.

.. testcode::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')


Create the parameterized circuit
================================

When creating the circuit you can include the :class:`~.Parameter` as if it was a defined number.

.. testcode::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

       ┌───────┐
    q: ┤ Rx(θ) ├
       └───────┘


Assign values to parameters
===========================

You can use these two methods to assign values to the :class:`~.Parameter`\ s in your circuit:

* :meth:`~.QuantumCircuit.bind_parameters` 
* :meth:`~.QuantumCircuit.assign_parameters` 

:meth:`~.QuantumCircuit.bind_parameters`
--------------------------------------------------------

In order to use this method, you have to specify either a dictionary of the form ``{parameter: value,...}`` or an iterable formed only by numeric values, that will be assigned following the order from :attr:`~.QuantumCircuit.parameters`.

.. testcode::

    import numpy as np

    theta_values = [0, np.pi/2, np.pi]
    qc_bind_list = [qc.bind_parameters({theta: theta_value}) for theta_value in theta_values]

    for i in range(3):
        print(qc_bind_list[i].draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

       ┌───────┐
    q: ┤ Rx(0) ├
       └───────┘
       ┌─────────┐
    q: ┤ Rx(π/2) ├
       └─────────┘
       ┌───────┐
    q: ┤ Rx(π) ├
       └───────┘

:meth:`~.QuantumCircuit.assign_parameters`
----------------------------------------------------------

This method works identically like :meth:`~.QuantumCircuit.bind_parameters`  except that you can also assign other :class:`~.Parameter` objects instead of only numbers to the :class:`~.Parameter`\ s in your circuit.

.. testcode::

    phi = Parameter('ϕ')

    theta_values = [np.pi/2, phi]
    qc_assign_list = [qc.assign_parameters({theta: theta_value}) for theta_value in theta_values]

    for i in range(2):
        print(qc_assign_list[i].draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

       ┌─────────┐
    q: ┤ Rx(π/2) ├
       └─────────┘
       ┌───────┐
    q: ┤ Rx(ϕ) ├
       └───────┘


Another difference between :meth:`~.QuantumCircuit.bind_parameters` and :meth:`~.QuantumCircuit.assign_parameters` is that for the latter, you can make it change your original circuit instead of creating a new one by setting the ``inplace`` argument to ``True``.

.. testcode::

    qc.assign_parameters({theta: np.pi/4}, inplace=True)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

       ┌─────────┐
    q: ┤ Rx(π/4) ├
       └─────────┘