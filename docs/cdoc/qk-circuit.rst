=========
QkCircuit
=========

.. code-block:: c

   typedef struct QkCircuit QkCircuit

The fundamental element of quantum computing is the *quantum circuit*.  This is a computational
routine that can be run, one shot at a time, on a quantum processing unit (QPU).  A circuit will act
on a predefined amount of quantum data (in Qiskit, we only directly support qubits) with unitary
operations (gates), measurements and resets.  In addition, a quantum circuit can contain operations
on classical data, including real-time computations and control-flow constructs, which are executed
by the controllers of the QPU. The ``QkCircuit`` struct  exposes a low
level interface to Qiskit's quantum circuit data structure and exposes
only what is defined in the inner data model of Qiskit. Therefore it
is missing some functionality that is available in the higher level
Python :class:`.QuantumCircuit` class.

Below is an example of a quantum circuit that makes a three-qubit Greenberger–Horne–Zeilinger (GHZ)
state defined as:

.. math::

   |\psi\rangle = \left( |000\rangle + |111\rangle \right) / \sqrt{2}

.. code-block:: c

    #include <qiskit.h>

    // Create a circuit with three qubits and 3 classical bits
    QkCircuit *circ = qk_circuit_new(3, 0);
    // H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
    qk_circuit_gate(qc, QkGate_h, {0}, NULL);
    // A CX (CNOT) gate on control qubit 0 and target qubit 1 generating a Bell state.
    qk_circuit_gate(qc, QkGate_cx, {0, 1}, NULL);
    // A CX (CNOT) gate on control qubit 0 and target qubit 2 generating a GHZ state.
    qk_circuit_gate(qc, QkGate_cx, {0, 1}, NULL);
    // Free the created circuit.
    qk_circuit_free(qc);

Functions
=========

.. doxygengroup:: QkCircuit
   :content-only:

