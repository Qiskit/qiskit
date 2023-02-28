
########################
Create a quantum circuit
########################

This guide shows how to initialize a quantum circuit.

There are two ways to create a :class:`~.QuantumCircuit` object:

* Specifying the number of qubits and bits.
* Creating :class:`~.QuantumRegister`\ s and :class:`~.ClassicalRegister`\ s.

Create from number of qubits and bits
=====================================

In order to create a :class:`~.QuantumCircuit` by only specifying the number of bits and qubits, you need to follow these steps.

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
     

If you don't want to include any classical bits, you don't have to write `QuantumCircuit(n_qubits,0)` but you can omit the number of classical bits.

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

In order to create a quantum register, you have to define a :class:`~.QuantumRegister` object, passing as argument the desired number of qubits.

.. testcode::

    from qiskit import QuantumRegister

    # Create QuantumRegister formed by 2 qubits
    qr1 = QuantumRegister(2)

    # Create QuantumRegister formed by 3 qubits
    qr2 = QuantumRegister(3)

Create classical registers
--------------------------

Analogously to the quantum registers, a classical register can be created by defining a :class:`~.ClassicalRegister` object, passing the number of classical bits as an argument.

.. testcode::

    from qiskit import ClassicalRegister

    # Create classical register with 2 classical bits
    cr1 = ClassicalRegister(2)

    # Create classical register with 1 classical bit
    cr2 = ClassicalRegister(1)

Initialize the quantum circuit
------------------------------

Now that you have defined the quantum and classical registers, you can define a :class:`~.QuantumCircuit` from them. Each register has to be introduced as a separate argument.

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
      

You can put the registers in any order, even mixing classical and quantum. However, the relative order of the :class:`~.QuantumRegister`\ s does affect the order of the qubits on the final circuit. In particular, the qubits from the first :class:`~.QuantumRegister` will be the first and so on. The same applies to the :class:`~.ClassicalRegister`\ s.

.. testcode::

    # Both the classical and quantum registers have the same relative order as in qc
    qc1 = QuantumCircuit(qr1, cr1, qr2, cr2)

    print(qc == qc1)

.. testoutput::

    True

.. testcode::

    # We change the order of the quantum registers
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
        
