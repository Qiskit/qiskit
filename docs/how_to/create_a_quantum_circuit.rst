
========================
Create a quantum circuit
========================

This guide shows how to initialize a quantum circuit.

There are two ways to create a :class:`~qiskit.circuit.QuantumCircuit` object:

* Specifying the number of qubits and bits.
* Creating :class:`~qiskit.circuit.QuantumRegister`\ s and :class:`~qiskit.circuit.ClassicalRegister`\ s

Create from number of qubits and bits
=====================================

In order to create a :class:`~qiskit.circuit.QuantumCircuit` by only specifying the number of bits and qubits, you need to follow these steps.

.. jupyter-execute::

    from qiskit import QuantumCircuit

    # Initialize number of qubits and classical bits
    n_qubits = 3
    n_bits = 2

    # Create the circuit
    qc = QuantumCircuit(n_qubits, n_bits)
    qc.draw()


If you don't want to include any classical bits, you don't have to write ``QuantumCircuit(n_qubits,0)`` but you can omit the number of classical bits.

.. jupyter-execute::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.draw()

Create from quantum and classical registers
===========================================

Create quantum registers
------------------------

In order to create a quantum register, you have to define a :class:`~qiskit.circuit.QuantumRegister` object, passing as argument the desired number of qubits.

.. jupyter-execute::

    from qiskit import QuantumRegister

    # Create QuantumRegister formed by 2 qubits
    qr1 = QuantumRegister(2)

    # Create QuantumRegister formed by 3 qubits
    qr2 = QuantumRegister(3)

Create classical registers
--------------------------

Analogously to the quantum registers, a classical register can be created by defining a :class:`~qiskit.circuit.ClassicalRegister` object, passing the number of classical bits as an argument.

.. jupyter-execute::

    from qiskit import ClassicalRegister

    # Create classical register with 2 classical bits
    cr1 = ClassicalRegister(2)

    # Create classical register with 1 classical bit
    cr2 = ClassicalRegister(1)

Initialize the quantum circuit
------------------------------

Now that you have defined the quantum and classical registers, you can define a :class:`~qiskit.circuit.QuantumCircuit` from them. Each register has to be introduced as a separate argument.

.. jupyter-execute::

    # Create the quantum circuit from the registers
    qc = QuantumCircuit(qr1, qr2, cr1, cr2)
    qc.draw()

You can put the registers in any order, even mixing classical and quantum. However, the relative order of the :class:`~qiskit.circuit.QuantumRegister`\ s does affect the order of the qubits on the final circuit. In particular, the qubits from the first :class:`~qiskit.circuit.QuantumRegister` will be the first and so on. The same applies to the :class:`~qiskit.circuit.ClassicalRegister`\ s.

.. jupyter-execute::

    # Both the classical and quantum registers have the same relative order as in qc
    qc1 = QuantumCircuit(qr1, cr1, qr2, cr2)

    qc == qc1


.. jupyter-execute::

    # We change the order of the quantum registers
    qc2 = QuantumCircuit(qr2, qr1, cr1, cr2)

    qc == qc2



.. jupyter-execute::

    qc2.draw()



.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_version_table
    %qiskit_copyright

