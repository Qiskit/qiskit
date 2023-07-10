#####################
Join quantum circuits
#####################

This guide shows you how to combine different :class:`~.QuantumCircuit` objects.

Create the circuits
===================

Let's first create two circuits, which will be joined together.

.. testcode::

    from qiskit import QuantumCircuit

    qc1 = QuantumCircuit(2,1)
    qc1.h(0)
    qc1.cx(0,1)
    qc1.measure(1,0)

    qc2 = QuantumCircuit(4,2)
    qc2.y(0)
    qc2.x(1)
    qc2.cx(0,3)
    qc2.measure(3,1)

    print(qc1.draw()) 
    print(qc2.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐        
    q_0: ┤ H ├──■─────
         └───┘┌─┴─┐┌─┐
    q_1: ─────┤ X ├┤M├
              └───┘└╥┘
    c: 1/═══════════╩═
                    0 
         ┌───┐        
    q_0: ┤ Y ├──■─────
         ├───┤  │     
    q_1: ┤ X ├──┼─────
         └───┘  │     
    q_2: ───────┼─────
              ┌─┴─┐┌─┐
    q_3: ─────┤ X ├┤M├
              └───┘└╥┘
    c: 2/═══════════╩═
                    1 

Join the circuits
=================

You can join the circuits together using these two methods:

1. Using the :meth:`~.QuantumCircuit.compose` method
2. Using the :meth:`~.QuantumCircuit.append` method

.. warning::

     The ``combine`` and ``extend`` methods have been deprecated in Qiskit Terra 0.17 and removed in 0.23. These methods are replaced by the :meth:`~.QuantumCircuit.compose` method which is more powerful. The removal of ``extend`` also means that the ``+`` and ``+=`` operators are no longer defined for :class:`~.QuantumCircuit`. Instead, you can use the ``&`` and ``&=`` operators respectively, which use :meth:`~.QuantumCircuit.compose`.

For both methods, if the two circuits being combined have different sizes, the method needs to be called in the circuit that is bigger (more qubits and more classical bits). 

Using the :meth:`~.QuantumCircuit.compose` method
-------------------------------------------------

In order to join two circuits with :meth:`~.QuantumCircuit.compose`, you only have to specify the circuit you want to insert. That way the qubits and bits of the smaller circuit will be included into the first qubits and bits of the bigger one in the original order they had. 

By default, :meth:`~.QuantumCircuit.compose` does not modify the original circuit to which it is applied but returns a new joint circuit object. This can be changed by setting the ``inplace`` argument to ``True``.

.. testcode::

    qc3 = qc2.compose(qc1)
    print(qc3.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐     ┌───┐        
    q_0: ┤ Y ├──■──┤ H ├──■─────
         ├───┤  │  └───┘┌─┴─┐┌─┐
    q_1: ┤ X ├──┼───────┤ X ├┤M├
         └───┘  │       └───┘└╥┘
    q_2: ───────┼─────────────╫─
              ┌─┴─┐ ┌─┐       ║ 
    q_3: ─────┤ X ├─┤M├───────╫─
              └───┘ └╥┘       ║ 
    c: 2/════════════╩════════╩═
                     1        0 

If you want to insert the qubits and bits into specific positions in the bigger circuit, you can use the ``qubits`` and ``bits`` arguments.

.. testcode::

    qc4 = qc2.compose(qc1, qubits=[3,1], clbits=[1])
    print(qc4.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐                     
    q_0: ┤ Y ├──■──────────────────
         ├───┤  │          ┌───┐┌─┐
    q_1: ┤ X ├──┼──────────┤ X ├┤M├
         └───┘  │          └─┬─┘└╥┘
    q_2: ───────┼────────────┼───╫─
              ┌─┴─┐┌─┐┌───┐  │   ║ 
    q_3: ─────┤ X ├┤M├┤ H ├──■───╫─
              └───┘└╥┘└───┘      ║ 
    c: 2/═══════════╩════════════╩═
                    1            1 

You can also join the smaller circuit in front of the bigger circuit by setting the ``front`` argument to ``True``.

.. testcode::

    qc5 = qc2.compose(qc1, front=True)
    print(qc5.draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐     ┌───┐             
    q_0: ┤ H ├──■──┤ Y ├───────■─────
         └───┘┌─┴─┐└┬─┬┘┌───┐  │     
    q_1: ─────┤ X ├─┤M├─┤ X ├──┼─────
              └───┘ └╥┘ └───┘  │     
    q_2: ────────────╫─────────┼─────
                     ║       ┌─┴─┐┌─┐
    q_3: ────────────╫───────┤ X ├┤M├
                     ║       └───┘└╥┘
    c: 2/════════════╩═════════════╩═
                     0             1 

Using the :meth:`~.QuantumCircuit.append` method
------------------------------------------------

In order to join two circuits with :meth:`~.QuantumCircuit.append`, you need to specify the circuit you want to add, as well as the qubits and classical bits (if there are any) onto which you want the circuit to be applied.

Different from :meth:`~.QuantumCircuit.compose`, this method modifies the circuit it is applied to, instead of returning a new circuit.

.. testcode::

    qc2.append(qc1, qargs=[3,1], cargs=[1])
    qc2.draw(cregbundle=False)

.. code-block:: text

         ┌───┐                        
    q_0: ┤ Y ├──■─────────────────────
         ├───┤  │     ┌──────────────┐
    q_1: ┤ X ├──┼─────┤1             ├
         └───┘  │     │              │
    q_2: ───────┼─────┤              ├
              ┌─┴─┐┌─┐│              │
    q_3: ─────┤ X ├┤M├┤0 circuit-101 ├
              └───┘└╥┘│              │
    c_0: ═══════════╬═╡              ╞
                    ║ │              │
    c_1: ═══════════╩═╡0             ╞
                      └──────────────┘

Unlike :meth:`~.QuantumCircuit.compose`, :meth:`~.QuantumCircuit.append` turns the smaller circuit into a single :class:`~qiskit.circuit.Instruction`. If you prefer joining the circuits using the individual gates, you can use :meth:`~.QuantumCircuit.decompose` to decompose the joint circuit.

.. testcode::

    print(qc2.decompose().draw())

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───────────────┐                     
    q_0: ┤ U3(π,π/2,π/2) ├──■──────────────────
         └─┬───────────┬─┘  │          ┌───┐┌─┐
    q_1: ──┤ U3(π,0,π) ├────┼──────────┤ X ├┤M├
           └───────────┘    │          └─┬─┘└╥┘
    q_2: ───────────────────┼────────────┼───╫─
                          ┌─┴─┐┌─┐┌───┐  │   ║ 
    q_3: ─────────────────┤ X ├┤M├┤ H ├──■───╫─
                          └───┘└╥┘└───┘      ║ 
    c: 2/═══════════════════════╩════════════╩═
                                1            1 
