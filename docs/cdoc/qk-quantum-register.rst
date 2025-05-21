=================
QkQuantumRegister
=================

.. code-block:: c

   typedef struct QkQuantumRegister QkQuantumRegister

A common way to instantiate several bits at once is to create a register, such
as by:

.. code-block: c

    #include <qiskit.h>

    QkQuantumRegister *qreg = qk_quantum_register_new(5, "my_qreg");

This has the advantage that you can give a collection of bits a name. This is
used as metadata around bits in the circuit when you add the register to the
circuit which will be used when exporting the circuit to interchange languages.
Continuing from the above example:

.. code-block: c

    QkCircuit *qc = qk_circuit_new(0, 0);
    qk_circuit_add_quantum_register(qc, qreg);
    uint32_t num_qubits = qk_circuit_num_qubits(qc); // 5

Circuits track registers, but registers themselves impart almost no behavioral
differences on circuits.

Functions
=========

.. doxygengroup:: QkQuantumRegister
   :members:
   :content-only:
