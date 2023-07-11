
########################
Create a quantum circuit
########################

This guide shows you how to create a quantum circuit.

There are two ways to create a :class:`~.QuantumCircuit` object:

1. Initialize :class:`~.QuantumCircuit` by directly specifying the number of qubits and classical bits you want.
2. Creating :class:`~.QuantumRegister`\ s and :class:`~.ClassicalRegister`\ s and use the registers to initialize a :class:`~.QuantumCircuit`

.. _create circuit with qubit numbers:

Create by specifying the number of qubits and classical bits
============================================================

You can create a :class:`~.QuantumCircuit` by only specifying the number of qubits and classical bits. For example:

.. testcode::

    from qiskit import QuantumCircuit

    # Initialize number of qubits and classical bits
    n_qubits = 3
    n_bits = 2

    # Create the circuit
    qc = QuantumCircuit(n_qubits, n_bits)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    q_0: 
        
    q_1: 
        
    q_2: 
        
    c: 2/
     

You can also create a :class:`~.QuantumCircuit` with only qubits and no classical bits, by omitting the number of classical bits:

.. testcode::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    q_0: 
        
    q_1: 
        
    q_2: 

Create from quantum and classical registers
===========================================

Create quantum registers
------------------------

You can create a :class:`~.QuantumRegister` object by passing the desired number of qubits as an argument:

.. testcode::

    from qiskit import QuantumRegister

    # Create a quantum register with 2 qubits
    qr1 = QuantumRegister(2)

    # Create a quantum register with 3 qubits
    qr2 = QuantumRegister(3)

Create classical registers
--------------------------

Similar to the quantum registers, you can create a :class:`~.ClassicalRegister` object by passing the desired number of classical bits as an argument:

.. testcode::

    from qiskit import ClassicalRegister

    # Create a classical register with 2 classical bits
    cr1 = ClassicalRegister(2)

    # Create a classical register with 1 classical bit
    cr2 = ClassicalRegister(1)

Create a quantum circuit
------------------------

Now that you have defined the quantum and classical registers, you can create a :class:`~.QuantumCircuit` with the registers: 

.. testcode::

    # Create the quantum circuit from the registers
    qc = QuantumCircuit(qr1, qr2, cr1, cr2)
    print(qc.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    q0_0: 
      
    q0_1: 
        
    q1_0: 
        
    q1_1: 
        
    q1_2: 
        
    c0: 2/
        
    c1: 1/
      

You can put the registers in any order, even mixing classical and quantum. However, the relative order of the :class:`~.QuantumRegister`\ s affects the order of the qubits in the final circuit. The qubits from the first :class:`~.QuantumRegister` will be the first and so on. The same applies to the :class:`~.ClassicalRegister`\ s.

.. testcode::

    # Resulting quantum circuits will be the same if the quantum and classical registers have the same relative order
    qc1 = QuantumCircuit(qr1, cr1, qr2, cr2)

    print(qc == qc1)

.. testoutput::

    True

.. testcode::

    # Resulting quantum circuits are different if the quantum or classical registers have different relative order
    qc2 = QuantumCircuit(qr2, qr1, cr1, cr2)

    print(qc == qc2)

.. testoutput::

    False


.. testcode::

    print(qc2.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    q1_0: 
      
    q1_1: 
        
    q1_2: 
        
    q0_0: 
        
    q0_1: 
        
    c0: 2/
        
    c1: 1/
        
