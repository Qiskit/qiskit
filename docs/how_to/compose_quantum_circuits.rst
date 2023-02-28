########################
Compose quantum circuits
########################

This guide shows how to combine different :class:`~.QuantumCircuit` objects.

Build the circuits
==================

The first step is creating the circuits you want to combine.

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

Combine the circuits
====================

Now that you have built the circuits, they can be combined with two different methods:

* :meth:`~.QuantumCircuit.compose`
* :meth:`~.QuantumCircuit.append`

One detail these two methods have in common is that if the circuits have different sizes, they have to be applied to the one that has the most of both qubits and classical bits.

:meth:`~.QuantumCircuit.compose`
------------------------------------------------

In order to combine two circuits with :meth:`~.QuantumCircuit.compose`, you only have to specify the circuit you want to insert. That way the qubits and bits of the smaller circuit will be included into the first qubits and bits of the bigger one in the original order they had. 

By default, :meth:`~.QuantumCircuit.compose` does not change the circuit to which it is applied but returns the composed circuit. This can be changed by setting the ``inplace`` argument to ``True``.

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

You can also apply the gates from the smaller circuit before those of the bigger one setting the ``front`` argument to ``True``.

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

:meth:`~.QuantumCircuit.append`
-----------------------------------------------

In order to combine two circuits with :meth:`~.QuantumCircuit.append`, you have to specify the circuit you want to add and the qubits and classical bits (if there are any) into which you want the circuit to be inserted.

This method changes the circuit to which it is applied instead of returning another one.

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

Unlike :meth:`~.QuantumCircuit.compose`, :meth:`~.QuantumCircuit.append` turns the smaller circuit into a single :class:`~qiskit.circuit.Instruction`, so in order to unroll it you can use :meth:`~.QuantumCircuit.decompose`

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
