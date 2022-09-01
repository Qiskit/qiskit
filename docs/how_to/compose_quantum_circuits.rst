========================
Compose quantum circuits
========================

This guide shows how to combine different :class:`~.QuantumCircuit` objects.

Build the circuits
==================

The first step is creating the circuits we want to combine.

.. jupyter-execute::

    from qiskit import QuantumCircuit
    qc1 = QuantumCircuit(2,1)
    qc1.h(0)
    qc1.cx(0,1)
    qc1.measure(0,0)

    qc2 = QuantumCircuit(4,2)
    qc2.y(1)
    qc2.measure(1,1)

Combine the circuits
====================

Now that we have built the circuits, they can be combined with two different methods:

* :meth:`~.QuantumCircuit.compose`
* :meth:`~.QuantumCircuit.append`

One detail these two methods have in common is that if the circuits have different sizes, they have to be applied to the one that has the most of both qubits and classical bits.

:meth:`~.QuantumCircuit.compose`
------------------------------------------------

In order to combine two circuits with :meth:`~.QuantumCircuit.compose`, you only have to specify the circuit you want to insert. That way the qubits and bits of the smaller circuit will be included into the first qubits and bits of the bigger one in the original order they had. 

By default, :meth:`~.QuantumCircuit.compose` does not change the circuit to which it is applied but returns the composed circuit. This can be changed by setting the ``inplace`` argument to ``True``.

.. jupyter-execute::

    qc3 = qc2.compose(qc1)
    qc3.draw()

If you want to insert the qubits and bits into specific positions in the bigger circuit, you can use the ``qubits`` and ``bits`` arguments.

.. jupyter-execute::

    qc4 = qc2.compose(qc1, qubits=[3,1], clbits=[1])
    qc4.draw()

You can also apply the gates from the smaller circuit before those of the bigger one setting the ``front`` argument to ``True``.

.. jupyter-execute::

    qc5 = qc2.compose(qc1, front=True)
    qc5.draw()

:meth:`~.QuantumCircuit.append`
-----------------------------------------------

In order to combine two circuits with :meth:`~.QuantumCircuit.append`, you have to specify the circuit you want to add and the qubits and classical bits (if there are any) into which you want the circuit to be inserted.

This method changes the circuit to which it is applied instead of returning another one.

.. jupyter-execute::

    qc2.append(qc1, qargs=[3,1], cargs=[1])
    qc2.draw(cregbundle=False)

Unlike :meth:`~.QuantumCircuit.compose`, :meth:`~.QuantumCircuit.append` turns the smaller circuit into a single :class:`~qiskit.circuit.Instruction`, so in order to unroll it you can use :meth:`~.QuantumCircuit.decompose`

.. jupyter-execute::

    qc2.decompose().draw()




