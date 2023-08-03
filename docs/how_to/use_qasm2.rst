################################
Import and export using QASM 2.0
################################

This guide shows you how to import and export quantum circuits using the ``QASM 2.0`` program.

QASM 2.0 stands for Quantum Assembly Language version 2.0, which is 
a programming language used to represent universal quantum computing based on 
the circuit model.

You can find more in its repository `OpenQASM 2.0 <https://github.com/openqasm/openqasm/tree/OpenQASM2.x>`_.

Design a quantum circuit
==============================

Build a 4 qubits quantum circuit with only CX,H, RZ gates and works with a custom quantum circuit using the method  :meth:`~qiskit.QuantumCircuit.to_gate`.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1) # custom  a X gate using H,Rz,H for this you need another QuantumCircuit.
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)
    gate = custom_gate.to_gate() # to_gate() convert the circuit custom_gate() in a gate that you need to indicate the posiiton in your quantum circuit.
    gate.name = 'custom X' # Put a name to the custom gate.

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])  # To add a custom gate you need to use the method append, where you indicate the gate and the QuantumReigsters.
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")


Export QuantumCircuit to OpenQASM 2
====================================


Parsing a quantum circuit in a OpenQASM 2 is support in :meth:`QuantumCircuit.qasm`, convert all in a string with all the information for default. 

.. testcode::

    qc.qasm()

.. testoutput::

    'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

Is possible formatted the string in a openqasm format, using the boolean parameter `formatted` being  its default value False, 
You ahould put in True to see the changes.

.. testcode::

    qc.qasm()

.. testoutput::

    OPENQASM 2.0;
    include "qelib1.inc";
    gate custom_X q0 { h q0; rz(pi) q0; h q0; }
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cx q[2],q[3];
    custom_X q[1];
    barrier q[0],q[1],q[2],q[3];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    measure q[3] -> c[3];


You can even save the code in a qasm file extension, you need to use the parameter ``filename``, this is a string you need to put the name following with ``.qasm``.


.. testcode::

    qc.qasm(filename='example.qasm')

.. testoutput::

    'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'




Import OpenQASM 2 to QuantumCircuit
====================================

Qiskit has a specific module that helps to import the OpenQASM2.0 files called qiskit-qasm2.


.. note::
    You can install the module using the following command

    ``pip install qiskit-qasm2``


Exist two methods you can use when you can read a string or a specific qasm file.
If you want to parse an OpenQASM 2 program from a string into a :class:`QuantumCircuit` you need :meth:`qiskit.qasm2.loads`


.. testcode::

    from qiskit import qasm2

    example = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'
    qc = qasm2.loads(example)
    qc.draw("mpl")


.. plot::

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1) 
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)
    gate = custom_gate.to_gate()
    gate.name = 'custom X'

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")

The case you have an OpenQASM 2 program from a file you need to use :meth:`qiskit.qasm2.load`.


.. testcode::

    qc = qasm2.loads('example.qasm')
    qc.draw("mpl")


.. plot::

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1) 
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)
    gate = custom_gate.to_gate()
    gate.name = 'custom X'

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")