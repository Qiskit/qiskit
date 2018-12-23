


Summary of Quantum Operations
=============================

In this section we will go into the different operations that are
available in Qiskit Terra. These are: - Single-qubit quantum gates -
Multi-qubit quantum gates - Measurements - Reset - Conditionals - State
initialization

We will also show you how to use the three different simulators: -
unitary_simulator - qasm_simulator - statevector_simulator

.. code:: ipython3

    # Useful additional packages 
    import matplotlib.pyplot as plt
    %matplotlib inline
    import numpy as np
    from math import pi

.. code:: ipython3

    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
    from qiskit.tools.visualization import circuit_drawer
    from qiskit.quantum_info import state_fidelity
    from qiskit import BasicAer
    
    backend = BasicAer.get_backend('unitary_simulator')

Single Qubit Quantum states
---------------------------

A single qubit quantum state can be written as

.. math:: |\psi\rangle = \alpha|0\rangle + \beta |1\rangle

where :math:`\alpha` and :math:`\beta` are complex numbers. In a
measurement the probability of the bit being in :math:`|0\rangle` is
:math:`|\alpha|^2` and :math:`|1\rangle` is :math:`|\beta|^2`. As a
vector this is

.. math::


   |\psi\rangle =  
   \begin{pmatrix}
   \alpha \\
   \beta
   \end{pmatrix}.

Note due to conservation probability :math:`|\alpha|^2+ |\beta|^2 = 1`
and since global phase is undetectable
:math:`|\psi\rangle := e^{i\delta} |\psi\rangle` we only requires two
real numbers to describe a single qubit quantum state.

A convenient representation is

.. math:: |\psi\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)e^{i\phi}|1\rangle

where :math:`0\leq \phi < 2\pi`, and :math:`0\leq \theta \leq \pi`. From
this it is clear that there is a one-to-one correspondence between qubit
states (:math:`\mathbb{C}^2`) and the points on the surface of a unit
sphere (:math:`\mathbb{R}^3`). This is called the Bloch sphere
representation of a qubit state.

Quantum gates/operations are usually represented as matrices. A gate
which acts on a qubit is represented by a :math:`2\times 2` unitary
matrix :math:`U`. The action of the quantum gate is found by multiplying
the matrix representing the gate with the vector which represents the
quantum state.

.. math:: |\psi'\rangle = U|\psi\rangle

A general unitary must be able to take the :math:`|0\rangle` to the
above state. That is

.. math::


   U = \begin{pmatrix}
   \cos(\theta/2) & a \\
   e^{i\phi}\sin(\theta/2) & b 
   \end{pmatrix}

where :math:`a` and :math:`b` are complex numbers constrained such that
:math:`U^\dagger U = I` for all :math:`0\leq\theta\leq\pi` and
:math:`0\leq \phi<2\pi`. This gives 3 constraints and as such
:math:`a\rightarrow -e^{i\lambda}\sin(\theta/2)` and
:math:`b\rightarrow e^{i\lambda+i\phi}\cos(\theta/2)` where
:math:`0\leq \lambda<2\pi` giving

.. math::


   U = \begin{pmatrix}
   \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
   e^{i\phi}\sin(\theta/2) & e^{i\lambda+i\phi}\cos(\theta/2) 
   \end{pmatrix}.

This is the most general form of a single qubit unitary.

Single-Qubit Gates
------------------

The single-qubit gates available are: - u gates - Identity gate - Pauli
gates - Clifford gates - :math:`C3` gates - Standard rotation gates

We have provided a backend: ``unitary_simulator`` to allow you to
calculate the unitary matrices.

.. code:: ipython3

    q = QuantumRegister(1)

u gates
~~~~~~~

In Qiskit we give you access to the general unitary using the :math:`u3`
gate

.. math::


   u3(\theta, \phi, \lambda) = U(\theta, \phi, \lambda) 

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.u3(pi/2,pi/2,pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌──────────────────────────┐
    q0_0: |0>┤ U3(1.5708,1.5708,1.5708) ├
             └──────────────────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 0.707+0.j   ,  0.   -0.707j],
           [ 0.   +0.707j, -0.707+0.j   ]])



The :math:`u2(\phi, \lambda) =u3(\pi/2, \phi, \lambda)` has the matrix
form

.. math::


   u2(\phi, \lambda) = 
   \frac{1}{\sqrt{2}} \begin{pmatrix}
   1 & -e^{i\lambda} \\
   e^{i\phi} & e^{i(\phi + \lambda)}
   \end{pmatrix}.

This is a useful gate as it allows us to create superpositions

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.u2(pi/2,pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───────────────────┐
    q0_0: |0>┤ U2(1.5708,1.5708) ├
             └───────────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 0.707+0.j   ,  0.   -0.707j],
           [ 0.   +0.707j, -0.707+0.j   ]])



The :math:`u1(\lambda)= u3(0, 0, \lambda)` gate has the matrix form

.. math::


   u1(\lambda) = 
   \begin{pmatrix}
   1 & 0 \\
   0 & e^{i \lambda}
   \end{pmatrix},

which is a useful as it allows us to apply a quantum phase.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.u1(pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────────────┐
    q0_0: |0>┤ U1(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+1.j]])



The :math:`u0(\delta)= u3(0, 0, 0)` gate is the identity matrix. It has
the matrix form

.. math::


   u0(\delta) = 
   \begin{pmatrix}
   1 & 0 \\
   0 & 1
   \end{pmatrix}.

The identity gate does nothing (but can add noise in the real device for
a period of time equal to fractions of the single qubit gate time)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.u0(pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────────────┐
    q0_0: |0>┤ U0(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]])



Identity gate
~~~~~~~~~~~~~

The identity gate is :math:`Id = u0(1)`.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.iden(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────┐
    q0_0: |0>┤ Id ├
             └────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]])



Pauli gates
~~~~~~~~~~~

:math:`X`: bit-flip gate
^^^^^^^^^^^^^^^^^^^^^^^^

The bit-flip gate :math:`X` is defined as:

.. math::


   X   =  
   \begin{pmatrix}
   0 & 1\\
   1 & 0
   \end{pmatrix}= u3(\pi,0,\pi)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.x(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ X ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j]])



:math:`Y`: bit- and phase-flip gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`Y` gate is defined as:

.. math::


   Y  = 
   \begin{pmatrix}
   0 & -i\\
   i & 0
   \end{pmatrix}=u3(\pi,\pi/2,\pi/2)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.y(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ Y ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[0.+0.j, 0.-1.j],
           [0.+1.j, 0.+0.j]])



:math:`Z`: phase-flip gate
^^^^^^^^^^^^^^^^^^^^^^^^^^

The phase flip gate :math:`Z` is defined as:

.. math::


   Z = 
   \begin{pmatrix}
   1 & 0\\
   0 & -1
   \end{pmatrix}=u1(\pi)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.z(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ Z ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 1.+0.j,  0.+0.j],
           [ 0.+0.j, -1.+0.j]])



Clifford gates
~~~~~~~~~~~~~~

Hadamard gate
^^^^^^^^^^^^^

.. math::


   H = 
   \frac{1}{\sqrt{2}}
   \begin{pmatrix}
   1 & 1\\
   1 & -1
   \end{pmatrix}= u2(0,\pi)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.h(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ H ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 0.707+0.j,  0.707+0.j],
           [ 0.707+0.j, -0.707+0.j]])



:math:`S` (or, :math:`\sqrt{Z}` phase) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   S = 
   \begin{pmatrix}
   1 & 0\\
   0 & i
   \end{pmatrix}= u1(\pi/2)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.s(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ S ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+1.j]])



:math:`S^{\dagger}` (or, conjugate of :math:`\sqrt{Z}` phase) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   S^{\dagger} = 
   \begin{pmatrix}
   1 & 0\\
   0 & -i
   \end{pmatrix}= u1(-\pi/2)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.sdg(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌─────┐
    q0_0: |0>┤ Sdg ├
             └─────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.-1.j]])



:math:`C3` gates
~~~~~~~~~~~~~~~~

:math:`T` (or, :math:`\sqrt{S}` phase) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   T = 
   \begin{pmatrix}
   1 & 0\\
   0 & e^{i \pi/4}
   \end{pmatrix}= u1(\pi/4) 

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.t(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐
    q0_0: |0>┤ T ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.   +0.j   , 0.   +0.j   ],
           [0.   +0.j   , 0.707+0.707j]])



:math:`T^{\dagger}` (or, conjugate of :math:`\sqrt{S}` phase) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   T^{\dagger} =  
   \begin{pmatrix}
   1 & 0\\
   0 & e^{-i \pi/4}
   \end{pmatrix}= u1(-pi/4)

They can be added as below.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.tdg(q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌─────┐
    q0_0: |0>┤ Tdg ├
             └─────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.   +0.j   , 0.   +0.j   ],
           [0.   +0.j   , 0.707-0.707j]])



Standard Rotations
~~~~~~~~~~~~~~~~~~

The standard rotation gates are those that define rotations around the
Paulis :math:`P=\{X,Y,Z\}`. They are defined as

.. math::  R_P(\theta) = \exp(-i \theta P/2) = \cos(\theta/2)I -i \sin(\theta/2)P

Rotation around X-axis
^^^^^^^^^^^^^^^^^^^^^^

.. math::


   R_x(\theta) = 
   \begin{pmatrix}
   \cos(\theta/2) & -i\sin(\theta/2)\\
   -i\sin(\theta/2) & \cos(\theta/2)
   \end{pmatrix} = u3(\theta, -\pi/2,\pi/2)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.rx(pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────────────┐
    q0_0: |0>┤ Rx(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[0.707+0.j   , 0.   -0.707j],
           [0.   -0.707j, 0.707+0.j   ]])



Rotation around Y-axis
^^^^^^^^^^^^^^^^^^^^^^

.. math::


   R_y(\theta) =
   \begin{pmatrix}
   \cos(\theta/2) & - \sin(\theta/2)\\
   \sin(\theta/2) & \cos(\theta/2).
   \end{pmatrix} =u3(\theta,0,0)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.ry(pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────────────┐
    q0_0: |0>┤ Ry(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 0.707+0.j, -0.707+0.j],
           [ 0.707+0.j,  0.707+0.j]])



Rotation around Z-axis
^^^^^^^^^^^^^^^^^^^^^^

.. math::


   R_z(\phi) = 
   \begin{pmatrix}
   e^{-i \phi/2} & 0 \\
   0 & e^{i \phi/2}
   \end{pmatrix}\equiv u1(\phi)

Note here we have used an equivalent as is different to u1 by global
phase :math:`e^{-i \phi/2}`.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.rz(pi/2,q)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌────────────┐
    q0_0: |0>┤ Rz(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+1.j]])



Note this is different due only to a global phase

Multi-Qubit Gates
-----------------

Mathematical Preliminaries
~~~~~~~~~~~~~~~~~~~~~~~~~~

The space of quantum computer grows exponential with the number of
qubits. For :math:`n` qubits the complex vector space has dimensions
:math:`d=2^n`. To describe states of a multi-qubit system, the tensor
product is used to “glue together” operators and basis vectors.

Let’s start by considering a 2-qubit system. Given two operators
:math:`A` and :math:`B` that each act on one qubit, the joint operator
:math:`A \otimes B` acting on two qubits is

.. math::

   \begin{equation}
       A\otimes B = 
       \begin{pmatrix} 
           A_{00} \begin{pmatrix} 
               B_{00} & B_{01} \\
               B_{10} & B_{11}
           \end{pmatrix} & A_{01}  \begin{pmatrix} 
                   B_{00} & B_{01} \\
                   B_{10} & B_{11}
               \end{pmatrix} \\
           A_{10}  \begin{pmatrix} 
                       B_{00} & B_{01} \\
                       B_{10} & B_{11}
                   \end{pmatrix} & A_{11}  \begin{pmatrix} 
                               B_{00} & B_{01} \\
                               B_{10} & B_{11}
                           \end{pmatrix}
       \end{pmatrix},                      
   \end{equation}

where :math:`A_{jk}` and :math:`B_{lm}` are the matrix elements of
:math:`A` and :math:`B`, respectively.

Analogously, the basis vectors for the 2-qubit system are formed using
the tensor product of basis vectors for a single qubit:

.. math::

   \begin{equation}\begin{split}
       |{00}\rangle &= \begin{pmatrix} 
           1 \begin{pmatrix} 
               1  \\
               0
           \end{pmatrix} \\
           0 \begin{pmatrix} 
               1  \\
               0 
           \end{pmatrix}
       \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \\ 0 \\0 \end{pmatrix}~~~|{01}\rangle = \begin{pmatrix} 
       1 \begin{pmatrix} 
       0 \\
       1
       \end{pmatrix} \\
       0 \begin{pmatrix} 
       0  \\
       1 
       \end{pmatrix}
       \end{pmatrix} = \begin{pmatrix}0 \\ 1 \\ 0 \\ 0 \end{pmatrix}\end{split}
   \end{equation}

.. math::

   \begin{equation}\begin{split}|{10}\rangle = \begin{pmatrix} 
       0\begin{pmatrix} 
       1  \\
       0
       \end{pmatrix} \\
       1\begin{pmatrix} 
       1 \\
       0 
       \end{pmatrix}
       \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}~~~   |{11}\rangle = \begin{pmatrix} 
       0 \begin{pmatrix} 
       0  \\
       1
       \end{pmatrix} \\
       1\begin{pmatrix} 
       0  \\
       1 
       \end{pmatrix}
       \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \\1 \end{pmatrix}\end{split}
   \end{equation}.

Note we’ve introduced a shorthand for the tensor product of basis
vectors, wherein :math:`|0\rangle \otimes |0\rangle` is written as
:math:`|00\rangle`. The state of an :math:`n`-qubit system can described
using the :math:`n`-fold tensor product of single-qubit basis vectors.
Notice that the basis vectors for a 2-qubit system are 4-dimensional; in
general, the basis vectors of an :math:`n`-qubit sytsem are
:math:`2^{n}`-dimensional, as noted earlier.

Basis vector ordering in Qiskit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within the physics community, the qubits of a multi-qubit systems are
typically ordered with the first qubit on the left-most side of the
tensor product and the last qubit on the right-most side. For instance,
if the first qubit is in state :math:`|0\rangle` and second is in state
:math:`|1\rangle`, their joint state would be :math:`|01\rangle`. Qiskit
uses a slightly different ordering of the qubits, in which the qubits
are represented from the most significant bit (MSB) on the left to the
least significant bit (LSB) on the right (big-endian). This is similar
to bitstring representation on classical computers, and enables easy
conversion from bitstrings to integers after measurements are performed.
For the example just given, the joint state would be represented as
:math:`|10\rangle`. Importantly, *this change in the representation of
multi-qubit states affects the way multi-qubit gates are represented in
Qiskit*, as discussed below.

The representation used in Qiskit enumerates the basis vectors in
increasing order of the integers they represent. For instance, the basis
vectors for a 2-qubit system would be ordered as :math:`|00\rangle`,
:math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle`. Thinking
of the basis vectors as bit strings, they encode the integers 0,1,2 and
3, respectively.

Controlled operations on qubits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common multi-qubit gate involves the application of a gate to one
qubit, conditioned on the state of another qubit. For instance, we might
want to flip the state of the second qubit when the first qubit is in
:math:`|0\rangle`. Such gates are known as *controlled gates*. The
standard multi-qubit gates consist of two-qubit gates and three-qubit
gates. The two-qubit gates are: - controlled Pauli gates - controlled
Hadamard gate - controlled rotation gates - controlled phase gate -
controlled u3 gate - swap gate

The three-qubit gates are: - Toffoli gate - Fredkin gate

Two-qubit gates
---------------

Most of the two-gates are of the controlled type (the SWAP gate being
the exception). In general, a controlled two-qubit gate :math:`C_{U}`
acts to apply the single-qubit unitary :math:`U` to the second qubit
when the state of the first qubit is in :math:`|1\rangle`. Suppose
:math:`U` has a matrix representation

.. math:: U = \begin{pmatrix} u_{00} & u_{01} \\ u_{10} & u_{11}\end{pmatrix}.

We can work out the action of :math:`C_{U}` as follows. Recall that the
basis vectors for a two-qubit system are ordered as
:math:`|00\rangle, |01\rangle, |10\rangle, |11\rangle`. Suppose the
**control qubit** is **qubit 0** (which, according to Qiskit’s
convention, is one the *right-hand* side of the tensor product). If the
control qubit is in :math:`|1\rangle`, :math:`U` should be applied to
the **target** (qubit 1, on the *left-hand* side of the tensor product).
Therefore, under the action of :math:`C_{U}`, the basis vectors are
transformed according to

.. math::

   \begin{align*}
   C_{U}: \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle} &\rightarrow \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle} &\rightarrow \underset{\text{qubit}~1}{U|0\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle} &\rightarrow \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle} &\rightarrow \underset{\text{qubit}~1}{U|1\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle}\\
   \end{align*}.

In matrix form, the action of :math:`C_{U}` is

.. math::

   \begin{equation}
       C_U = \begin{pmatrix}
       1 & 0 & 0 & 0 \\
       0 & u_{00} & 0 & u_{01} \\
       0 & 0 & 1 & 0 \\
       0 & u_{10} &0 & u_{11}
           \end{pmatrix}.
   \end{equation}

To work out these matrix elements, let

.. math:: C_{(jk), (lm)} = \left(\underset{\text{qubit}~1}{\langle j |} \otimes \underset{\text{qubit}~0}{\langle k |}\right) C_{U} \left(\underset{\text{qubit}~1}{| l \rangle} \otimes \underset{\text{qubit}~0}{| k \rangle}\right),

compute the action of :math:`C_{U}` (given above), and compute the inner
products.

As shown in the examples below, this operation is implemented in Qiskit
as ``cU(q[0],q[1])``.

If **qubit 1 is the control and qubit 0 is the target**, then the basis
vectors are transformed according to

.. math::

   \begin{align*}
   C_{U}: \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle} &\rightarrow \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle} &\rightarrow \underset{\text{qubit}~1}{|0\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{|0\rangle} &\rightarrow \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{U|0\rangle}\\
   C_{U}: \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{|1\rangle} &\rightarrow \underset{\text{qubit}~1}{|1\rangle}\otimes \underset{\text{qubit}~0}{U|1\rangle}\\
   \end{align*},

which implies the matrix form of :math:`C_{U}` is

.. math::

   \begin{equation}
       C_U = \begin{pmatrix}
       1 & 0 & 0  & 0 \\
       0 & 1 & 0 & 0 \\
       0 & 0 & u_{00} & u_{01} \\
       0 & 0 & u_{10} & u_{11}
           \end{pmatrix}.
   \end{equation}

.. code:: ipython3

    q = QuantumRegister(2)

Controlled Pauli Gates
~~~~~~~~~~~~~~~~~~~~~~

Controlled-X (or, controlled-NOT) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The controlled-not gate flips the ``target`` qubit when the control
qubit is in the state :math:`|1\rangle`. If we take the MSB as the
control qubit (e.g. ``cx(q[1],q[0])``), then the matrix would look like

.. math::


   C_X = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 0 & 1\\
   0 & 0 & 1 & 0
   \end{pmatrix}. 

However, when the LSB is the control qubit, (e.g. ``cx(q[0],q[1])``),
this gate is equivalent to the following matrix:

.. math::


   C_X = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 0 & 0 & 1\\
   0 & 0 & 1 & 0\\
   0 & 1 & 0 & 0
   \end{pmatrix}. 

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cx(q[0],q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">              
    q1_0: |0>──■──
             ┌─┴─┐
    q1_1: |0>┤ X ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])



Controlled :math:`Y` gate
^^^^^^^^^^^^^^^^^^^^^^^^^

Apply the :math:`Y` gate to the target qubit if the control qubit is the
MSB

.. math::


   C_Y = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 0 & -i\\
   0 & 0 & i & 0
   \end{pmatrix},

or when the LSB is the control

.. math::


   C_Y = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 0 & 0 & -i\\
   0 & 0 & 1 & 0\\
   0 & i & 0 & 0
   \end{pmatrix}.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cy(q[0],q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">              
    q1_0: |0>──■──
             ┌─┴─┐
    q1_1: |0>┤ Y ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]])



Controlled :math:`Z` (or, controlled Phase) gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, the controlled Z gate flips the phase of the target qubit if
the control qubit is :math:`1`. The matrix looks the same regardless of
whether the MSB or LSB is the control qubit:

.. math::


   C_Z = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 1 & 0\\
   0 & 0 & 0 & -1
   \end{pmatrix}

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cz(q[0],q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">            
    q1_0: |0>─■─
              │ 
    q1_1: |0>─■─
                </pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])



Controlled Hadamard gate
~~~~~~~~~~~~~~~~~~~~~~~~

Apply :math:`H` gate to the target qubit if the control qubit is
:math:`|1\rangle`. Below is the case where the control is the LSB qubit.

.. math::


   C_H = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}}\\
   0 & 0 & 1 & 0\\
   0 & \frac{1}{\sqrt{2}}  & 0& -\frac{1}{\sqrt{2}}
   \end{pmatrix}

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.ch(q[0],q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">              
    q1_0: |0>──■──
             ┌─┴─┐
    q1_1: |0>┤ H ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 0.707+0.707j,  0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.5  +0.5j  ,  0.   +0.j   ,  0.5  +0.5j  ],
           [ 0.   +0.j   ,  0.   +0.j   ,  0.707+0.707j,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.5  +0.5j  ,  0.   +0.j   , -0.5  -0.5j  ]])



Controlled rotation gates
~~~~~~~~~~~~~~~~~~~~~~~~~

Controlled rotation around Z-axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform rotation around Z-axis on the target qubit if the control qubit
(here LSB) is :math:`|1\rangle`.

.. math::


   C_{Rz}(\lambda) = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & e^{-i\lambda/2} & 0 & 0\\
   0 & 0 & 1 & 0\\
   0 & 0 & 0 & e^{i\lambda/2}
   \end{pmatrix}

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.crz(pi/2,q[0],q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">                       
    q1_0: |0>──────■───────
             ┌─────┴──────┐
    q1_1: |0>┤ Rz(1.5708) ├
             └────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
           [0.   +0.j   , 0.707-0.707j, 0.   +0.j   , 0.   +0.j   ],
           [0.   +0.j   , 0.   +0.j   , 1.   +0.j   , 0.   +0.j   ],
           [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.707+0.707j]])



Controlled phase rotation
~~~~~~~~~~~~~~~~~~~~~~~~~

Perform a phase rotation if both qubits are in the :math:`|11\rangle`
state. The matrix looks the same regardless of whether the MSB or LSB is
the control qubit.

.. math::


   C_{u1}(\lambda) = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 1 & 0\\
   0 & 0 & 0 & e^{i\lambda}
   \end{pmatrix}

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cu1(pi/2,q[0], q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">                  
    q1_0: |0>─■───────
              │1.5708 
    q1_1: |0>─■───────
                      </pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j]])



Controlled :math:`u3` rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform controlled-\ :math:`u3` rotation on the target qubit if the
control qubit (here LSB) is :math:`|1\rangle`.

.. math::


   C_{u3}(\theta, \phi, \lambda) \equiv 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & e^{-i(\phi+\lambda)/2}\cos(\theta/2) & 0 & -e^{-i(\phi-\lambda)/2}\sin(\theta/2)\\
   0 & 0 & 1 & 0\\
   0 & e^{i(\phi-\lambda)/2}\sin(\theta/2) & 0 & e^{i(\phi+\lambda)/2}\cos(\theta/2)
   \end{pmatrix}.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cu3(pi/2, pi/2, pi/2, q[0], q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">                                     
    q1_0: |0>─────────────■──────────────
             ┌────────────┴─────────────┐
    q1_1: |0>┤ U3(1.5708,1.5708,1.5708) ├
             └──────────────────────────┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[ 1.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   -0.707j,  0.   +0.j   , -0.707+0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   ,  1.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.707+0.j   ,  0.   +0.j   ,  0.   +0.707j]])



SWAP gate
~~~~~~~~~

The SWAP gate exchanges the two qubits. It transforms the basis vectors
as

.. math:: |00\rangle \rightarrow |00\rangle~,~|01\rangle \rightarrow |10\rangle~,~|10\rangle \rightarrow |01\rangle~,~|11\rangle \rightarrow |11\rangle,

which gives a matrix representation of the form

.. math::


   \mathrm{SWAP} = 
   \begin{pmatrix}
   1 & 0 & 0 & 0\\
   0 & 0 & 1 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 0 & 1
   \end{pmatrix}.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.swap(q[0], q[1])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">            
    q1_0: |0>─X─
              │ 
    q1_1: |0>─X─
                </pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])



Three-qubit gates
-----------------

There are two commonly-used three-qubit gates. For three qubits, the
basis vectors are ordered as

.. math:: |000\rangle, |001\rangle, |010\rangle, |011\rangle, |100\rangle, |101\rangle, |110\rangle, |111\rangle,

which, as bitstrings, represent the integers :math:`0,1,2,\cdots, 7`.
Again, Qiskit uses a representation in which the first qubit is on the
right-most side of the tensor product and the third qubit is on the
left-most side:

.. math:: |abc\rangle : \underset{\text{qubit 2}}{|a\rangle}\otimes \underset{\text{qubit 1}}{|b\rangle}\otimes \underset{\text{qubit 0}}{|c\rangle}.

Toffoli gate (:math:`ccx` gate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Toffoli
gate <https://en.wikipedia.org/wiki/Quantum_logic_gate#Toffoli_(CCNOT)_gate>`__
flips the third qubit if the first two qubits (LSB) are both
:math:`|1\rangle`:

.. math:: |abc\rangle \rightarrow |bc\oplus a\rangle \otimes |b\rangle \otimes c \rangle.

In matrix form, the Toffoli gate is

.. math::


   C_{CX} = 
   \begin{pmatrix}
   1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
   0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
   \end{pmatrix}.

.. code:: ipython3

    q = QuantumRegister(3)

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.ccx(q[0], q[1], q[2])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">              
    q2_0: |0>──■──
               │  
    q2_1: |0>──■──
             ┌─┴─┐
    q2_2: |0>┤ X ├
             └───┘</pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])



Controlled swap gate (Fredkin Gate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Fredkin
gate <https://en.wikipedia.org/wiki/Quantum_logic_gate#Fredkin_(CSWAP)_gate>`__,
or the *controlled swap gate*, exchanges the second and third qubits if
the first qubit (LSB) is :math:`|1\rangle`:

.. math::  |abc\rangle \rightarrow \begin{cases} |bac\rangle~~\text{if}~c=1 \cr |abc\rangle~~\text{if}~c=0 \end{cases}.

In matrix form, the Fredkin gate is

.. math::


   C_{\mathrm{SWAP}} = 
   \begin{pmatrix}
   1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
   0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{pmatrix}.

.. code:: ipython3

    qc = QuantumCircuit(q)
    qc.cswap(q[0], q[1], q[2])
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">            
    q2_0: |0>─■─
              │ 
    q2_1: |0>─X─
              │ 
    q2_2: |0>─X─
                </pre>



.. code:: ipython3

    job = execute(qc, backend)
    job.result().get_unitary(qc, decimals=3)




.. parsed-literal::

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])



Non unitary operations
----------------------

Now we have gone through all the unitary operations in quantum circuits
we also have access to non-unitary operations. These include
measurements, reset of qubits, and classical conditional operations.

.. code:: ipython3

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

Measurements
~~~~~~~~~~~~

We don’t have access to all the information when we make a measurement
in a quantum computer. The quantum state is projected onto the standard
basis. Below are two examples showing a circuit that is prepared in a
basis state and the quantum computer prepared in a superposition state.

.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.measure(q, c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌─┐
    q3_0: |0>┤M├
             └╥┘
     c0_0: 0 ═╩═
                </pre>



.. code:: ipython3

    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'0': 1024}



The simulator predicts that 100 percent of the time the classical
register returns 0.

.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.h(q)
    qc.measure(q, c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐┌─┐
    q3_0: |0>┤ H ├┤M├
             └───┘└╥┘
     c0_0: 0 ══════╩═
                     </pre>



.. code:: ipython3

    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'0': 502, '1': 522}



The simulator predicts that 50 percent of the time the classical
register returns 0 or 1.

Reset
~~~~~

It is also possible to ``reset`` qubits to the :math:`|0\rangle` state
in the middle of computation. Note that ``reset`` is not a Gate
operation, since it is irreversible.

.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.reset(q[0])
    qc.measure(q, c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">              ┌─┐
    q3_0: |0>─|0>─┤M├
                  └╥┘
     c0_0: 0 ══════╩═
                     </pre>



.. code:: ipython3

    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'0': 1024}



.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.h(q)
    qc.reset(q[0])
    qc.measure(q, c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐     ┌─┐
    q3_0: |0>┤ H ├─|0>─┤M├
             └───┘     └╥┘
     c0_0: 0 ═══════════╩═
                          </pre>



.. code:: ipython3

    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'0': 1024}



Here we see that for both of these circuits the simulator always
predicts that the output is 100 percent in the 0 state.

Conditional operations
~~~~~~~~~~~~~~~~~~~~~~

It is also possible to do operations conditioned on the state of the
classical register

.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.x(q[0]).c_if(c, 0)
    qc.measure(q,c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌─────┐┌─┐
    q3_0: |0>┤  X  ├┤M├
             ├──┴──┤└╥┘
     c0_0: 0 ╡ = 0 ╞═╩═
             └─────┘   </pre>



Here the classical bit always takes the value 0 so the qubit state is
always flipped.

.. code:: ipython3

    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'1': 1024}



.. code:: ipython3

    qc = QuantumCircuit(q, c)
    qc.h(q)
    qc.measure(q,c)
    qc.x(q[0]).c_if(c, 0)
    qc.measure(q,c)
    qc.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">         ┌───┐┌─┐┌─────┐┌─┐
    q3_0: |0>┤ H ├┤M├┤  X  ├┤M├
             └───┘└╥┘├──┴──┤└╥┘
     c0_0: 0 ══════╩═╡ = 0 ╞═╩═
                     └─────┘   </pre>



.. code:: ipython3

    job = execute(qc, backend, shots=1024)
    job.result().get_counts(qc)




.. parsed-literal::

    {'1': 1024}



Here the classical bit by the first measurement is random but the
conditional operation results in the qubit being deterministically put
into :math:`|1\rangle`.

Arbitrary initialization
~~~~~~~~~~~~~~~~~~~~~~~~

What if we want to initialize a qubit register to an arbitrary state? An
arbitrary state for :math:`n` qubits may be specified by a vector of
:math:`2^n` amplitudes, where the sum of amplitude-norms-squared equals
1. For example, the following three-qubit state can be prepared:

.. math:: |\psi\rangle = \frac{i}{4}|000\rangle + \frac{1}{\sqrt{8}}|001\rangle + \frac{1+i}{4}|010\rangle + \frac{1+2i}{\sqrt{8}}|101\rangle + \frac{1}{4}|110\rangle

.. code:: ipython3

    # Initializing a three-qubit quantum state
    import math
    desired_vector = [
        1 / math.sqrt(16) * complex(0, 1),
        1 / math.sqrt(8) * complex(1, 0),
        1 / math.sqrt(16) * complex(1, 1),
        0,
        0,
        1 / math.sqrt(8) * complex(1, 2),
        1 / math.sqrt(16) * complex(1, 0),
        0]
    
    
    q = QuantumRegister(3)
    
    qc = QuantumCircuit(q)
    
    qc.initialize(desired_vector, [q[0],q[1],q[2]])




.. parsed-literal::

    <qiskit.extensions.quantum_initializer._initializer.InitializeGate at 0xa1ed7fd30>



.. code:: ipython3

    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    qc_state = job.result().get_statevector(qc)
    qc_state 




.. parsed-literal::

    array([0.25      +0.j        , 0.        -0.35355339j,
           0.25      -0.25j      , 0.        +0.j        ,
           0.        +0.j        , 0.70710678-0.35355339j,
           0.        -0.25j      , 0.        +0.j        ])



`Fidelity <https://en.wikipedia.org/wiki/Fidelity_of_quantum_states>`__
is useful to check whether two states are same or not. For quantum
(pure) states :math:`\left|\psi_1\right\rangle` and
:math:`\left|\psi_2\right\rangle`, the fidelity is

.. math::


   F\left(\left|\psi_1\right\rangle,\left|\psi_2\right\rangle\right) = \left|\left\langle\psi_1\middle|\psi_2\right\rangle\right|^2.

The fidelity is equal to :math:`1` if and only if two states are same.

.. code:: ipython3

    state_fidelity(desired_vector,qc_state)




.. parsed-literal::

    1.0



Further details:
^^^^^^^^^^^^^^^^

How does the desired state get generated behind the scenes? There are
multiple methods for doing this. Qiskit uses a `method proposed by
Shende et al <https://arxiv.org/abs/quant-ph/0406176>`__. Here, the idea
is to assume the quantum register to have started from our desired
state, and construct a circuit that takes it to the
:math:`|00..0\rangle` state. The initialization circuit is then the
reverse of such circuit.

To take an arbitrary quantum state to the zero state in the
computational basis, we perform an iterative procedure that disentangles
qubits from the register one-by-one. We know that any arbitrary
single-qubit state :math:`|\rho\rangle` can be taken to the
:math:`|0\rangle` state using a :math:`\phi`-degree rotation about the Z
axis followed by a :math:`\theta`-degree rotation about the Y axis:

.. math:: R_y(-\theta)R_z(-\phi)|\rho\rangle = re^{it}|0\rangle

Since now we are dealing with :math:`n` qubits instead of just 1, we
must factorize the state vector to separate the Least Significant Bit
(LSB):

.. math::

   \begin{align*}
    |\psi\rangle =& \alpha_{0_0}|00..00\rangle + \alpha_{0_1}|00..01\rangle + \alpha_{1_0}|00..10\rangle + \alpha_{1_1}|00..11\rangle + ... \\&+ \alpha_{(2^{n-1}-1)_0}|11..10\rangle + \alpha_{(2^{n-1}-1)_1}|11..11\rangle \\
   =& |00..0\rangle (\alpha_{0_0}|0\rangle + \alpha_{0_1}|1\rangle) + |00..1\rangle (\alpha_{1_0}|0\rangle + \alpha_{1_1}|1\rangle) + ... \\&+ |11..1\rangle (\alpha_{(2^{n-1}-1)_0}(|0\rangle + \alpha_{(2^{n-1}-1)_1}|1\rangle) \\
   =& |00..0\rangle|\rho_0\rangle + |00..1\rangle|\rho_1\rangle + ... + |11..1\rangle|\rho_{2^{n-1}-1}\rangle
   \end{align*}

Now each of the single-qubit states
:math:`|\rho_0\rangle, ..., |\rho_{2^{n-1}-1}\rangle` can be taken to
:math:`|0\rangle` by finding appropriate :math:`\phi` and :math:`\theta`
angles per the equation above. Doing this simultaneously on all states
amounts to the following unitary, which disentangles the LSB:

.. math::

   U = \begin{pmatrix} 
   R_{y}(-\theta_0)R_{z}(-\phi_0) & & & &\\  
   & R_{y}(-\theta_1)R_{z}(-\phi_1) & & &\\
   & . & & &\\
   & & . & &\\
   & & & & R_y(-\theta_{2^{n-1}-1})R_z(-\phi_{2^{n-1}-1})
   \end{pmatrix} 

Hence,

.. math:: U|\psi\rangle = \begin{pmatrix} r_0e^{it_0}\\ r_1e^{it_1}\\ . \\ . \\ r_{2^{n-1}-1}e^{it_{2^{n-1}-1}} \end{pmatrix}\otimes|0\rangle

U can be implemented as a “quantum multiplexor” gate, since it is a
block diagonal matrix. In the quantum multiplexor formalism, a block
diagonal matrix of size :math:`2^n \times 2^n`, and consisting of
:math:`2^s` blocks, is equivalent to a multiplexor with :math:`s` select
qubits and :math:`n-s` data qubits. Depending on the state of the select
qubits, the corresponding blocks are applied to the data qubits. A
multiplexor of this kind can be implemented after recursive
decomposition to primitive gates of cx, rz and ry.
