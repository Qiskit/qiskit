#########################
Order of qubits in Qiskit
#########################

While most physics textbooks represent an :math:`n`-qubit system as the tensor product :math:`Q_0\otimes Q_1 \otimes ... \otimes Q_{n-1}`, where :math:`Q_j` is the :math:`j^{\mathrm{th}}` qubit, Qiskit uses the inverse order, that is, :math:`Q_{n-1}\otimes \dotsb \otimes Q_1 \otimes Q_{0}`. As explained in `this video <https://www.youtube.com/watch?v=EiqHj3_Avps>`_ from `Qiskit's YouTube channel <https://www.youtube.com/@qiskit>`_, this is done to follow the convention in classical computing, in which the :math:`n^{\mathrm{th}}` bit or most significant bit (MSB) is placed on the left while the least significant bit (LSB) is placed on the right. This ordering convention is called little-endian while the one from the physics textbooks is called big-endian.

This means that if we have, for example, a 3-qubit system with qubit 0 in state :math:`|1\rangle` and qubits 1 and 2 in state :math:`|0\rangle`, Qiskit would label this state as :math:`|001\rangle` while most physics textbooks would label this state as :math:`|100\rangle`. 

The matrix representation of any multi-qubit gate is also affected by this different qubit ordering. For example, if we consider the single-qubit gate

.. math::

    U = \begin{pmatrix} u_{00} & u_{01} \\ u_{10} & u_{11} \end{pmatrix}

And we want a controlled version :math:`C_U` whose control qubit is qubit 0 and whose target is qubit 1, following Qiskit's little-endian convention its matrix representation would be

.. math::

    C_U = \begin{pmatrix} 1 & 0 & 0 & 0 \\0 & u_{00} & 0 & u_{01} \\ 0 & 0 & 1 & 0 \\ 0 & u_{10} & 0& u_{11} \end{pmatrix}

while it would be written as this following big-endian convention:

.. math::

    C_U = \begin{pmatrix} 1 & 0 & 0 & 0 \\0 & 1 & 0 & 0 \\ 0 & 0 & u_{00} & u_{01} \\ 0 & 0 & u_{10} & u_{11} \end{pmatrix}


For more details about how the ordering of qubits affects the matrix representation of any particular gate, check its entry in the circuit :mod:`~qiskit.circuit.library`.

This different order can also make the circuit corresponding to an algorithm from a textbook a bit more complicated to visualize. Fortunately, Qiskit provides a way to represent a :class:`~.QuantumCircuit` with the most significant qubits on top, just like in the textbooks. This can be done by setting the ``reverse_bits`` argument of the :meth:`~.QuantumCircuit.draw` method to ``True``.

Let's try this for a 3-qubit Quantum Fourier Transform (:class:`~.QFT`).

.. plot::
    :include-source:
    :context:

    from qiskit.circuit.library import QFT

    qft = QFT(3)
    qft.decompose().draw('mpl')

.. plot::
    :include-source:
    :context: close-figs

    qft.decompose().draw('mpl', reverse_bits=True)