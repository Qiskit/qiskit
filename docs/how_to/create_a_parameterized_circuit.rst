==============================
Create a parameterized circuit
==============================

This guide will show how to create a :class:`~qiskit.circuit.QuantumCircuit` that includes parameters.

Define the parameters
=====================

In order to define a parameter, you need to create a :class:`~qiskit.circuit.Parameter` object. To do that you only need to choose a ``name``, that can be any unicode string like ``'θ'``.

.. jupyter-execute::

    from qiskit.circuit import Parameter

    theta = Parameter('θ')

Create the parameterized circuit
================================

When creating the circuit you can include the :class:`~qiskit.circuit.Parameter` as if it was a defined number.

.. jupyter-execute::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.draw('mpl')

Assign values to parameters
===========================

You can use these two methods to assign values to the :class:`~qiskit.circuit.Parameter`\ s in your circuit:

* :meth:`~qiskit.circuit.QuantumCircuit.bind_parameters()` 
* :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters()` 

:meth:`~qiskit.circuit.QuantumCircuit.bind_parameters()`
-------------------------------------------------------

In order to use this method, you have to specify either a dictionary of the form ``{parameter: value,...}`` or an iterable formed only by numeric values, that will be assigned following the order from :attr:`~qiskit.circuit.QuantumCircuit.parameters`.

.. jupyter-execute::

    import numpy as np

    theta_values = [0, np.pi/2, np.pi]
    qc_bind_list = [qc.bind_parameters({theta: theta_value}) for theta_value in theta_values]

    for i in range(3):
        display(qc_bind_list[i].draw('mpl'))

:meth:`~qiskit.circuit.QuantumCircuit.assign_parameters()`
---------------------------------------------------------

This method works identically like :meth:`~qiskit.circuit.QuantumCircuit.bind_parameters()`  except that you can also assign other :class:`~qiskit.circuit.Parameter` objects instead of only numbers to the :class:`~qiskit.circuit.Parameter`\ s in your circuit.

.. jupyter-execute::

    phi = Parameter('ϕ')

    theta_values = [np.pi/2, phi]
    qc_assign_list = [qc.assign_parameters({theta: theta_value}) for theta_value in theta_values]

    for i in range(2):
        display(qc_assign_list[i].draw('mpl'))

Another difference between :meth:`~qiskit.circuit.QuantumCircuit.bind_parameters()` and :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters()` is that for the latter, you can make it change your original circuit instead of creating a new one by setting the ``inplace`` argument to ``True``.

.. jupyter-execute::

    qc.assign_parameters({theta: np.pi/4}, inplace=True)
    qc.draw('mpl')


.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_version_table
    %qiskit_copyright
