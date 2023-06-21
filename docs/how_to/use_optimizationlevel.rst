#####################################
Using optimization_level on transpile
#####################################

This guide shows you how to obtain different versions of a general
quantum circuit using :meth:`~qiskit.compiler.transpile` and its
parameter  :attr:`~qiskit.transpile.optimization_level`.
`Transpile <https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html>`__
is a method that helps you in converting a quantum circuit, while
considering the limitations and structures of a specific backend. For
example, it follows a particular basis gates, coupling map, initial
layout or other parameters. The attribute  :attr:`~qiskit.transpile.optimization_level` helps
you optimize your quantum circuit. Its value is integer and can be
between 0 through 3, where the higher number indicates better optimize
results. You can find more information about this parameter
`here <https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html#Preset-Pass-Managers>`__.

    .. code-block::

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuito #fake model of ibmq_quito

The first step at the moment use :meth:`~qiskit.compiler.transpile`, is to select a quantum
proccessor. This can be either a `real quantum
proccesor <https://quantum-computing.ibm.com/services/resources>`__ or a
`fake
model <https://qiskit.org/documentation/apidoc/providers_fake_provider.html>`__.

.. testcode::

    # For this example  is used the quito Fake model
    backend = FakeQuito()

Initialize the quantum circuit
==============================


When you work with a quantum circuit coposed of other quantum gates, for
example `Controlled
SWAP <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.cswap.html>`__
gate, which is a three-qubit gate, you can decompose it into one and
two-qubit gates.

.. testcode::

    qr = QuantumRegister(3) #init a quantum register with 3 qubits
    qc = QuantumCircuit(qr) # init the quantum circuit with 3 qubits 
    
    qc.cswap(qr[0],qr[1],qr[2]) #add to the quantum circuit the cswap gate
    
    print(qc) # plot the quantum circuit


.. testoutput::

             
    q0_0: ─■─
           │ 
    q0_1: ─X─
           │ 
    q0_2: ─X─
             


Using basis gates
=================

In :meth:`~qiskit.compiler.transpile` you can indicate different parameters as is
:attr:`~qiskit.compiler.transpile.basis_gates`, which is a list of strings that indicates the quantum
gates that use the backend selected.

In ``ibmq_quito`` its basis gates shows as

.. testcode::

    backend.configuration().basis_gates




.. testoutput::

    ['id', 'rz', 'sx', 'x', 'cx', 'reset']



You need add it in the parameter :attr:`~qiskit.compiler.transpile.basis_gates`.

.. testcode::

    qc_bg = transpile(qc,basis_gates = backend.configuration().basis_gates)
    
    print(qc_bg)


.. testoutput::

    global phase: 5π/8
                                                                                 »
    q0_0: ────────────────────────────────────────────────────■──────────────────»
          ┌───┐                                               │                  »
    q0_1: ┤ X ├──────────────────────────────■────────────────┼───────────────■──»
          └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─┴─┐»
    q0_2: ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├»
               └─────────┘└────┘└─────────┘└───┘└──────────┘└───┘└─────────┘└───┘»
    «                                   ┌─────────┐                 
    «q0_0: ──────────────■───────■──────┤ Rz(π/4) ├──────■──────────
    «      ┌─────────┐   │     ┌─┴─┐    ├─────────┴┐   ┌─┴─┐   ┌───┐
    «q0_1: ┤ Rz(π/4) ├───┼─────┤ X ├────┤ Rz(-π/4) ├───┤ X ├───┤ X ├
    «      ├─────────┴┐┌─┴─┐┌──┴───┴───┐└──┬────┬──┘┌──┴───┴──┐└─┬─┘
    «q0_2: ┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├───┤ √X ├───┤ Rz(π/2) ├──■──
    «      └──────────┘└───┘└──────────┘   └────┘   └─────────┘     


Optimization level in basis gates
---------------------------------

When you apply the :attr:`~qiskit.transpile.optimization_level` is 0, means no optimization.
The result is the decompose of your circuit with no optimization.

.. testcode::

    qc_bg0 = transpile(qc,basis_gates = backend.configuration().basis_gates,optimization_level = 0)
    
    print(qc_bg0)


.. testoutput::

    global phase: 5π/8
                                                                                 »
    q0_0: ────────────────────────────────────────────────────■──────────────────»
          ┌───┐                                               │                  »
    q0_1: ┤ X ├──────────────────────────────■────────────────┼───────────────■──»
          └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─┴─┐»
    q0_2: ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├»
               └─────────┘└────┘└─────────┘└───┘└──────────┘└───┘└─────────┘└───┘»
    «                                  ┌─────────┐                       
    «q0_0: ──────────────■───────■─────┤ Rz(π/4) ├───■───────────────────
    «      ┌─────────┐   │     ┌─┴─┐   ├─────────┴┐┌─┴─┐            ┌───┐
    «q0_1: ┤ Rz(π/4) ├───┼─────┤ X ├───┤ Rz(-π/4) ├┤ X ├────────────┤ X ├
    «      ├─────────┴┐┌─┴─┐┌──┴───┴──┐├─────────┬┘├───┴┐┌─────────┐└─┬─┘
    «q0_2: ┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├┤ Rz(π/2) ├─┤ √X ├┤ Rz(π/2) ├──■──
    «      └──────────┘└───┘└─────────┘└─────────┘ └────┘└─────────┘     


The case :attr:`~qiskit.transpile.optimization_level` is 1, is the default value and do a
light optimization in the quantum circuit. In the no optimizate circuit
exist the gates :math:`RZ(\pi/4)` and :math:`RZ(\pi/2)` can be convert
into only one gate, :math:`RZ(3\pi/4)`.

.. testcode::

    qc_bg1 = transpile(qc,basis_gates=backend.configuration().basis_gates,optimization_level = 1)
    
    print(qc_bg1)


.. testoutput::

    global phase: 5π/8
                                                                                 »
    q0_0: ────────────────────────────────────────────────────■──────────────────»
          ┌───┐                                               │                  »
    q0_1: ┤ X ├──────────────────────────────■────────────────┼───────────────■──»
          └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─┴─┐»
    q0_2: ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├»
               └─────────┘└────┘└─────────┘└───┘└──────────┘└───┘└─────────┘└───┘»
    «                                   ┌─────────┐                 
    «q0_0: ──────────────■───────■──────┤ Rz(π/4) ├──────■──────────
    «      ┌─────────┐   │     ┌─┴─┐    ├─────────┴┐   ┌─┴─┐   ┌───┐
    «q0_1: ┤ Rz(π/4) ├───┼─────┤ X ├────┤ Rz(-π/4) ├───┤ X ├───┤ X ├
    «      ├─────────┴┐┌─┴─┐┌──┴───┴───┐└──┬────┬──┘┌──┴───┴──┐└─┬─┘
    «q0_2: ┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├───┤ √X ├───┤ Rz(π/2) ├──■──
    «      └──────────┘└───┘└──────────┘   └────┘   └─────────┘     


If :attr:`~qiskit.transpile.optimization_level` is 2, exist a medium optimization in the
quantum circuit. Depends of the complexity of the quantum circuits that
could be the same result as in 1.

.. testcode::

    qc_bg2 = transpile(qc,basis_gates = backend.configuration().basis_gates,optimization_level = 2)
    
    print(qc_bg2)


.. testoutput::

    global phase: 5π/8
                                                                                 »
    q0_0: ────────────────────────────────────────────────────■──────────────────»
          ┌───┐                                               │                  »
    q0_1: ┤ X ├──────────────────────────────■────────────────┼───────────────■──»
          └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─┴─┐»
    q0_2: ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├»
               └─────────┘└────┘└─────────┘└───┘└──────────┘└───┘└─────────┘└───┘»
    «                                   ┌─────────┐                 
    «q0_0: ──────────────■───────■──────┤ Rz(π/4) ├──────■──────────
    «      ┌─────────┐   │     ┌─┴─┐    ├─────────┴┐   ┌─┴─┐   ┌───┐
    «q0_1: ┤ Rz(π/4) ├───┼─────┤ X ├────┤ Rz(-π/4) ├───┤ X ├───┤ X ├
    «      ├─────────┴┐┌─┴─┐┌──┴───┴───┐└──┬────┬──┘┌──┴───┴──┐└─┬─┘
    «q0_2: ┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├───┤ √X ├───┤ Rz(π/2) ├──■──
    «      └──────────┘└───┘└──────────┘   └────┘   └─────────┘     


The :attr:`~qiskit.transpile.optimization_level` is 3, exist a heavy optimization in the
quantum circuit. This configuration take more time and try to check the
optimal solution for the circuit consider the basis gates, as result you
can see reduce for one
`CX <https://qiskit.org/documentation/stubs/qiskit.circuit.library.CXGate.html>`__
gate and adding eight of one qubit gate.

.. testcode::

    qc_bg3 = transpile(qc,basis_gates = backend.configuration().basis_gates,optimization_level = 3)
    
    print(qc_bg3)


.. testoutput::

    global phase: 3π/8
                                                                                »
    q0_0: ──────────────────────────────────────────────────────────────────────»
             ┌────────┐  ┌────┐ ┌────────┐        ┌────┐                        »
    q0_1: ───┤ Rz(-π) ├──┤ √X ├─┤ Rz(-π) ├──■─────┤ √X ├────────────────────────»
          ┌──┴────────┴─┐├────┤┌┴────────┤┌─┴─┐┌──┴────┴─┐┌────┐┌──────────────┐»
    q0_2: ┤ Rz(-2.3821) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(π/2) ├┤ √X ├┤ Rz(-0.75949) ├»
          └─────────────┘└────┘└─────────┘└───┘└─────────┘└────┘└──────────────┘»
    «                                                                         »
    «q0_0: ───────────────────■────────────────────────────────■───────■──────»
    «                         │                  ┌─────────┐   │     ┌─┴─┐    »
    «q0_1: ───────────────────┼───────────────■──┤ Rz(π/4) ├───┼─────┤ X ├────»
    «      ┌────┐┌─────────┐┌─┴─┐┌─────────┐┌─┴─┐├─────────┴┐┌─┴─┐┌──┴───┴───┐»
    «q0_2: ┤ √X ├┤ Rz(π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├»
    «      └────┘└─────────┘└───┘└─────────┘└───┘└──────────┘└───┘└──────────┘»
    «      ┌─────────┐                 
    «q0_0: ┤ Rz(π/4) ├──────■──────────
    «      ├─────────┴┐   ┌─┴─┐   ┌───┐
    «q0_1: ┤ Rz(-π/4) ├───┤ X ├───┤ X ├
    «      └──┬────┬──┘┌──┴───┴──┐└─┬─┘
    «q0_2: ───┤ √X ├───┤ Rz(π/2) ├──■──
    «         └────┘   └─────────┘     


Using init layout
=================

One interesting parameter is :attr:`~qiskit.transpile.init_layout`, it could be a dict or list
where you can assign a :class:`~qiskit.QuantumRegister` variable to a physical qubit.

One example you can do it conser the next layout of 3 qubits where the
key is the register and the value is the physical qubit.

.. testcode::

    initial_layout = {qr[0]: 0, #assign the qr[0] to the physical qubit 0
     qr[1]: 3,   #assign the qr[1] to the physical qubit 3
     qr[2]: 4}  #assign the qr[2] to the physical qubit 4
    
    initial_layout


.. testoutput::

    {Qubit(QuantumRegister(3, 'q0'), 0): 0,
     Qubit(QuantumRegister(3, 'q0'), 1): 3,
     Qubit(QuantumRegister(3, 'q0'), 2): 4}



You need add it in the parameter :attr:`~qiskit.transpile.init_layout`. You must consider that
ancilla qubitsare the qubits that your circuit is not using and the
number next to the arrow is the index of the qubit.

.. testcode::

    qc_il = transpile(qc,initial_layout = initial_layout)
    
    print(qc_il)


.. testoutput::

                                                                    »
         q0_0 -> 0 ────────────────────────■─────────────────────■──»
                                           │                     │  »
    ancilla_0 -> 1 ────────────────────────┼─────────────────────┼──»
                                           │                     │  »
    ancilla_1 -> 2 ────────────────────────┼─────────────────────┼──»
                   ┌───┐                   │             ┌───┐   │  »
         q0_1 -> 3 ┤ X ├───────■───────────┼─────────■───┤ T ├───┼──»
                   └─┬─┘┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐»
         q0_2 -> 4 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                        └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘»
    «                                   ┌───┐           
    «     q0_0 -> 0 ─────────■──────────┤ T ├───■───────
    «                        │          └───┘   │       
    «ancilla_0 -> 1 ─────────┼──────────────────┼───────
    «                        │                  │       
    «ancilla_1 -> 2 ─────────┼──────────────────┼───────
    «                      ┌─┴─┐       ┌─────┐┌─┴─┐┌───┐
    «     q0_1 -> 3 ───────┤ X ├───────┤ Tdg ├┤ X ├┤ X ├
    «               ┌──────┴───┴──────┐└─────┘└───┘└─┬─┘
    «     q0_2 -> 4 ┤ U3(π/2,0,-3π/4) ├──────────────■──
    «               └─────────────────┘                 


Optimization level in init layout
---------------------------------

The result consider the decompose of the circuit without optimization
and using the physical qubits zero,three and four. Is similar result
with respect basis_gates, the difference is that the qubits are you
using is 0,3,4 and 1,2 they put as ancillas.

.. testcode::

    qc_il0 = transpile(qc,initial_layout = initial_layout, optimization_level = 0)
    
    qc_il0.draw("text")




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">                                                                      ┌───┐ »
         q0_0 -> 0 ────────────────────────■─────────────────────■────■───┤ T ├─»
                                           │                     │    │   └───┘ »
    ancilla_0 -> 1 ────────────────────────┼─────────────────────┼────┼─────────»
                                           │                     │    │         »
    ancilla_1 -> 2 ────────────────────────┼─────────────────────┼────┼─────────»
                   ┌───┐                   │             ┌───┐   │  ┌─┴─┐┌─────┐»
         q0_1 -> 3 ┤ X ├───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├»
                   └─┬─┘┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐├───┤└┬───┬┘»
         q0_2 -> 4 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├─»
                        └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘ »
    «                         
    «     q0_0 -> 0 ──■───────
    «                 │       
    «ancilla_0 -> 1 ──┼───────
    «                 │       
    «ancilla_1 -> 2 ──┼───────
    «               ┌─┴─┐┌───┐
    «     q0_1 -> 3 ┤ X ├┤ X ├
    «               └───┘└─┬─┘
    «     q0_2 -> 4 ───────■──
    «                         </pre>



The default version use the configuration equals to 1, being the same
circuit but one
`Hadamard <https://qiskit.org/documentation/stubs/qiskit.circuit.library.HGate.html>`__
gate, adding
`:math:`U_3(\pi/2,0,-3\pi/4)` <https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html>`__.

.. testcode::

    qc_il1 = transpile(qc,initial_layout = initial_layout, optimization_level = 1)
    
    print(qc_il1)


.. testoutput::

                                                                    »
         q0_0 -> 0 ────────────────────────■─────────────────────■──»
                                           │                     │  »
    ancilla_0 -> 1 ────────────────────────┼─────────────────────┼──»
                                           │                     │  »
    ancilla_1 -> 2 ────────────────────────┼─────────────────────┼──»
                   ┌───┐                   │             ┌───┐   │  »
         q0_1 -> 3 ┤ X ├───────■───────────┼─────────■───┤ T ├───┼──»
                   └─┬─┘┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐»
         q0_2 -> 4 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                        └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘»
    «                                   ┌───┐           
    «     q0_0 -> 0 ─────────■──────────┤ T ├───■───────
    «                        │          └───┘   │       
    «ancilla_0 -> 1 ─────────┼──────────────────┼───────
    «                        │                  │       
    «ancilla_1 -> 2 ─────────┼──────────────────┼───────
    «                      ┌─┴─┐       ┌─────┐┌─┴─┐┌───┐
    «     q0_1 -> 3 ───────┤ X ├───────┤ Tdg ├┤ X ├┤ X ├
    «               ┌──────┴───┴──────┐└─────┘└───┘└─┬─┘
    «     q0_2 -> 4 ┤ U3(π/2,0,-3π/4) ├──────────────■──
    «               └─────────────────┘                 


The medium optimization in the quantum circuit, use a short number of
gates the quantum circuits could be the same result as in 1.

.. testcode::

    qc_il2 = transpile(qc,initial_layout = initial_layout, optimization_level = 2)
    
    print(qc_il2)


.. testoutput::

                                                                    »
         q0_0 -> 0 ────────────────────────■─────────────────────■──»
                                           │                     │  »
    ancilla_0 -> 1 ────────────────────────┼─────────────────────┼──»
                                           │                     │  »
    ancilla_1 -> 2 ────────────────────────┼─────────────────────┼──»
                   ┌───┐                   │             ┌───┐   │  »
         q0_1 -> 3 ┤ X ├───────■───────────┼─────────■───┤ T ├───┼──»
                   └─┬─┘┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐»
         q0_2 -> 4 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                        └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘»
    «                                   ┌───┐           
    «     q0_0 -> 0 ─────────■──────────┤ T ├───■───────
    «                        │          └───┘   │       
    «ancilla_0 -> 1 ─────────┼──────────────────┼───────
    «                        │                  │       
    «ancilla_1 -> 2 ─────────┼──────────────────┼───────
    «                      ┌─┴─┐       ┌─────┐┌─┴─┐┌───┐
    «     q0_1 -> 3 ───────┤ X ├───────┤ Tdg ├┤ X ├┤ X ├
    «               ┌──────┴───┴──────┐└─────┘└───┘└─┬─┘
    «     q0_2 -> 4 ┤ U3(π/2,0,-3π/4) ├──────────────■──
    «               └─────────────────┘                 


The heavy optimization in the quantum circuit, reduce two cx gates, a
hadamard gate an a
`T\ :math:`^\dagger` <https://qiskit.org/documentation/stubs/qiskit.circuit.library.TdgGate.html>`__
in a unitary gate. And using the same qubits you indicate in
``init_layout``.

.. testcode::

    qc_il3 = transpile(qc,initial_layout = initial_layout, optimization_level = 3)
    
    print(qc_il3)


.. testoutput::

                                                                             »
         q0_0 -> 0 ──────────────■─────────────────────■───────────■─────────»
                                 │                     │           │         »
    ancilla_0 -> 1 ──────────────┼─────────────────────┼───────────┼─────────»
                                 │                     │           │         »
    ancilla_1 -> 2 ──────────────┼─────────────────────┼───────────┼─────────»
                   ┌──────────┐  │             ┌───┐   │         ┌─┴─┐       »
         q0_1 -> 3 ┤0         ├──┼─────────■───┤ T ├───┼─────────┤ X ├───────»
                   │  Unitary │┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐┌──────┴───┴──────┐»
         q0_2 -> 4 ┤1         ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ U3(π/2,0,-3π/4) ├»
                   └──────────┘└───┘└───┘└───┘└─────┘└───┘└─────────────────┘»
    «                ┌───┐           
    «     q0_0 -> 0 ─┤ T ├───■───────
    «                └───┘   │       
    «ancilla_0 -> 1 ─────────┼───────
    «                        │       
    «ancilla_1 -> 2 ─────────┼───────
    «               ┌─────┐┌─┴─┐┌───┐
    «     q0_1 -> 3 ┤ Tdg ├┤ X ├┤ X ├
    «               └─────┘└───┘└─┬─┘
    «     q0_2 -> 4 ──────────────■──
    «                                


Using coupling map
==================

Other parameter you can use in transpile is :attr:`~qiskit.transpile.coupling_map`, which is a
list, must be given as an adjacency matrix, where each entry specifies
all directed two-qubit interactions supported by backend.

In ``ibmq_quito`` its coupling map is

.. testcode::

    backend.configuration().coupling_map




.. testoutput::

    [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]



You need add it in the parameter :attr:`~qiskit.transpile.coupling_map`. The connections is
using the following quantum circuit is **[[1,3],[3,4],[4,3]]**.

.. testcode::

    qc_cm = transpile(qc,coupling_map = backend.configuration().coupling_map)
    
    print(qc_cm)


.. testoutput::

                                                                    »
    ancilla_0 -> 0 ─────────────────────────────────────────────────»
                        ┌───┐┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐┌───┐»
         q0_2 -> 1 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                     │  └───┘└─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└─────┘└─┬─┘»
         q0_0 -> 2 ──┼─────────┼───────────■─────────┼───────────■──»
                   ┌─┴─┐       │                     │   ┌───┐      »
         q0_1 -> 3 ┤ X ├───────■─────────────────────■───┤ T ├──────»
                   └───┘                                 └───┘      »
    ancilla_1 -> 4 ─────────────────────────────────────────────────»
                                                                    »
    «                                                           
    «ancilla_0 -> 0 ────────────────────────────────────────────
    «               ┌─────────────────┐   ┌───┐┌─────┐┌───┐┌───┐
    «     q0_2 -> 1 ┤ U3(π/2,0,-3π/4) ├─X─┤ X ├┤ Tdg ├┤ X ├┤ X ├
    «               └─────────────────┘ │ └─┬─┘└┬───┬┘└─┬─┘└─┬─┘
    «     q0_0 -> 2 ────────────────────┼───■───┤ T ├───■────┼──
    «                                   │       └───┘        │  
    «     q0_1 -> 3 ────────────────────X────────────────────■──
    «                                                           
    «ancilla_1 -> 4 ────────────────────────────────────────────
    «                                                           


Optimization level in coupling map
----------------------------------

The result consider the best gates to design a decompose of your circuit
with no optimization and following the connections **[[0, 1], [1, 0],
[1, 2], [2, 1]]**

.. testcode::

    qc_cm0 = transpile(qc,coupling_map = backend.configuration().coupling_map,optimization_level = 0)
    
    print(qc_cm0)


.. testoutput::

                                                                               »
         q0_0 -> 0 ───────────────────────────■─────────────────────■───────■──»
                   ┌───┐                    ┌─┴─┐┌───┐┌───┐┌─────┐┌─┴─┐   ┌─┴─┐»
         q0_1 -> 1 ┤ X ├───────■──────────X─┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├─X─┤ X ├»
                   └─┬─┘┌───┐┌─┴─┐┌─────┐ │ └───┘└───┘└─┬─┘└┬───┬┘└───┘ │ ├───┤»
         q0_2 -> 2 ──■──┤ H ├┤ X ├┤ Tdg ├─X─────────────■───┤ T ├───────X─┤ T ├»
                        └───┘└───┘└─────┘                   └───┘         └───┘»
    ancilla_0 -> 3 ────────────────────────────────────────────────────────────»
                                                                               »
    ancilla_1 -> 4 ────────────────────────────────────────────────────────────»
                                                                               »
    «                ┌───┐           
    «     q0_0 -> 0 ─┤ T ├───■───────
    «               ┌┴───┴┐┌─┴─┐┌───┐
    «     q0_1 -> 1 ┤ Tdg ├┤ X ├┤ X ├
    «               └┬───┬┘└───┘└─┬─┘
    «     q0_2 -> 2 ─┤ H ├────────■──
    «                └───┘           
    «ancilla_0 -> 3 ─────────────────
    «                                
    «ancilla_1 -> 4 ─────────────────
    «                                


The default version use the configuration equals to 1, being the same
circuit but with four
`SWAP <https://qiskit.org/documentation/stubs/qiskit.circuit.library.SwapGate.html>`__
gates less, and similar configuration that :attr:`~qiskit.transpile.init_layout` with
:attr:`~qiskit.transpile.optimization_level` = 1 and use the following connections **[[1, 3],
[2, 1], [3, 1]]**

.. testcode::

    qc_cm1 = transpile(qc,coupling_map = backend.configuration().coupling_map,optimization_level = 1)
    
    print(qc_cm1)


.. testoutput::

                                                                    »
    ancilla_0 -> 0 ─────────────────────────────────────────────────»
                        ┌───┐┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐┌───┐»
         q0_2 -> 1 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                     │  └───┘└─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└─────┘└─┬─┘»
         q0_0 -> 2 ──┼─────────┼───────────■─────────┼───────────■──»
                   ┌─┴─┐       │                     │   ┌───┐      »
         q0_1 -> 3 ┤ X ├───────■─────────────────────■───┤ T ├──────»
                   └───┘                                 └───┘      »
    ancilla_1 -> 4 ─────────────────────────────────────────────────»
                                                                    »
    «                                                           
    «ancilla_0 -> 0 ────────────────────────────────────────────
    «               ┌─────────────────┐   ┌───┐┌─────┐┌───┐┌───┐
    «     q0_2 -> 1 ┤ U3(π/2,0,-3π/4) ├─X─┤ X ├┤ Tdg ├┤ X ├┤ X ├
    «               └─────────────────┘ │ └─┬─┘└┬───┬┘└─┬─┘└─┬─┘
    «     q0_0 -> 2 ────────────────────┼───■───┤ T ├───■────┼──
    «                                   │       └───┘        │  
    «     q0_1 -> 3 ────────────────────X────────────────────■──
    «                                                           
    «ancilla_1 -> 4 ────────────────────────────────────────────
    «                                                           


The medium optimization in the quantum circuit, use a short number of
gates the quantum circuits could be the same result as in 1. And is not
always using the same qubits, in this case use the connections **[[0,
1], [1, 3], [3, 1]]**

.. testcode::

    qc_cm2 = transpile(qc,coupling_map = backend.configuration().coupling_map,optimization_level = 2)
    
    print(qc_cm2)


.. testoutput::

                                                                    »
    ancilla_0 -> 0 ─────────────────────────────────────────────────»
                        ┌───┐┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐┌───┐»
         q0_2 -> 1 ──■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├»
                   ┌─┴─┐└───┘└─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└┬───┬┘└─┬─┘»
         q0_1 -> 2 ┤ X ├───────■───────────┼─────────■───┤ T ├───┼──»
                   └───┘                   │             └───┘   │  »
         q0_0 -> 3 ────────────────────────■─────────────────────■──»
                                                                    »
    ancilla_1 -> 4 ─────────────────────────────────────────────────»
                                                                    »
    «                                                           
    «ancilla_0 -> 0 ────────────────────────────────────────────
    «               ┌─────────────────┐   ┌───┐┌─────┐┌───┐┌───┐
    «     q0_2 -> 1 ┤ U3(π/2,0,-3π/4) ├─X─┤ X ├┤ Tdg ├┤ X ├┤ X ├
    «               └─────────────────┘ │ └─┬─┘└─────┘└─┬─┘└─┬─┘
    «     q0_1 -> 2 ────────────────────X───┼───────────┼────■──
    «                                       │   ┌───┐   │       
    «     q0_0 -> 3 ────────────────────────■───┤ T ├───■───────
    «                                           └───┘           
    «ancilla_1 -> 4 ────────────────────────────────────────────
    «                                                           


The heavy optimization in the quantum circuit,is the same result of
:attr:`~qiskit.transpile.init_layout` when its :attr:`~qiskit.transpile.optimization_level` = 3, but this use the
following connections **[[1, 3], [3, 1],[4,3]]**

.. testcode::

    qc_cm3 = transpile(qc,coupling_map = backend.configuration().coupling_map,optimization_level = 3)
    
    print(qc_cm3)


.. testoutput::

                   ┌──────────┐                ┌───┐                            »
         q0_1 -> 0 ┤0         ├────────────■───┤ T ├──────────────────────────X─»
                   │  Unitary │┌───┐┌───┐┌─┴─┐┌┴───┴┐┌───┐┌─────────────────┐ │ »
         q0_2 -> 1 ┤1         ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ U3(π/2,0,-3π/4) ├─X─»
                   └──────────┘└─┬─┘└───┘└───┘└─────┘└─┬─┘└─────────────────┘   »
    ancilla_0 -> 2 ──────────────┼─────────────────────┼────────────────────────»
                                 │                     │                        »
         q0_0 -> 3 ──────────────■─────────────────────■────────────────────────»
                                                                                »
    ancilla_1 -> 4 ─────────────────────────────────────────────────────────────»
                                                                                »
    «                                     
    «     q0_1 -> 0 ───────────────────■──
    «               ┌───┐┌─────┐┌───┐┌─┴─┐
    «     q0_2 -> 1 ┤ X ├┤ Tdg ├┤ X ├┤ X ├
    «               └─┬─┘└─────┘└─┬─┘└───┘
    «ancilla_0 -> 2 ──┼───────────┼───────
    «                 │   ┌───┐   │       
    «     q0_0 -> 3 ──■───┤ T ├───■───────
    «                     └───┘           
    «ancilla_1 -> 4 ──────────────────────
    «                                     


Using backend’s information
===========================

When you apply :meth:`~qiskit.transpile` you can indicate different parameters of
your backend what has its particular properties, and is a collection of
the previous parameter in one object.

.. testcode::

    qc_b = transpile(qc,backend = backend)
    
    print(qc_b)


.. testoutput::

    global phase: 5π/8
                   ┌───┐                                                  »
         q0_1 -> 0 ┤ X ├──────────────────────────────■───────────────────»
                   └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌───┐»
         q0_2 -> 1 ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├»
                        └─────────┘└────┘└─────────┘└───┘└──────────┘└─┬─┘»
         q0_0 -> 2 ────────────────────────────────────────────────────■──»
                                                                          »
    ancilla_0 -> 3 ───────────────────────────────────────────────────────»
                                                                          »
    ancilla_1 -> 4 ───────────────────────────────────────────────────────»
                                                                          »
    «                               ┌─────────┐                                   »
    «     q0_1 -> 0 ─────────────■──┤ Rz(π/4) ├───────────────────────────────────»
    «               ┌─────────┐┌─┴─┐├─────────┴┐┌───┐┌──────────┐┌────┐┌─────────┐»
    «     q0_2 -> 1 ┤ Rz(π/4) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├┤ √X ├┤ Rz(π/2) ├»
    «               └─────────┘└───┘└──────────┘└─┬─┘└──────────┘└────┘└─────────┘»
    «     q0_0 -> 2 ──────────────────────────────■───────────────────────────────»
    «                                                                             »
    «ancilla_0 -> 3 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «ancilla_1 -> 4 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «                    ┌───┐                                
    «     q0_1 -> 0 ──■──┤ X ├──■──────────────────────────■──
    «               ┌─┴─┐└─┬─┘┌─┴─┐┌───┐┌──────────┐┌───┐┌─┴─┐
    «     q0_2 -> 1 ┤ X ├──■──┤ X ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ X ├
    «               └───┘     └───┘└─┬─┘├─────────┬┘└─┬─┘└───┘
    «     q0_0 -> 2 ─────────────────■──┤ Rz(π/4) ├───■───────
    «                                   └─────────┘           
    «ancilla_0 -> 3 ──────────────────────────────────────────
    «                                                         
    «ancilla_1 -> 4 ──────────────────────────────────────────
    «                                                         


If :attr:`~qiskit.transpile.optimization_level` is equals to 0,this shows the couplan map in
**[[0,1],[1,0],[1,2],[2,1]]**

.. testcode::

    qc_b0 = transpile(qc,backend = backend,optimization_level = 0)
    
    print(qc_b0)


.. testoutput::

    global phase: 5π/8
                                                                               »
         q0_0 -> 0 ────────────────────────────────────────────────────────────»
                   ┌───┐                                                  ┌───┐»
         q0_1 -> 1 ┤ X ├──────────────────────────────■────────────────■──┤ X ├»
                   └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐└─┬─┘»
         q0_2 -> 2 ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├──■──»
                        └─────────┘└────┘└─────────┘└───┘└──────────┘└───┘     »
    ancilla_0 -> 3 ────────────────────────────────────────────────────────────»
                                                                               »
    ancilla_1 -> 4 ────────────────────────────────────────────────────────────»
                                                                               »
    «                                                                         »
    «     q0_0 -> 0 ───────■────────────────────────────────■─────────────────»
    «                    ┌─┴─┐┌─────────┐┌───┐┌──────────┐┌─┴─┐     ┌───┐     »
    «     q0_1 -> 1 ──■──┤ X ├┤ Rz(π/4) ├┤ X ├┤ Rz(-π/4) ├┤ X ├──■──┤ X ├──■──»
    «               ┌─┴─┐└───┘└─────────┘└─┬─┘├─────────┬┘└───┘┌─┴─┐└─┬─┘┌─┴─┐»
    «     q0_2 -> 2 ┤ X ├──────────────────■──┤ Rz(π/4) ├──────┤ X ├──■──┤ X ├»
    «               └───┘                     └─────────┘      └───┘     └───┘»
    «ancilla_0 -> 3 ──────────────────────────────────────────────────────────»
    «                                                                         »
    «ancilla_1 -> 4 ──────────────────────────────────────────────────────────»
    «                                                                         »
    «                          ┌─────────┐                       
    «     q0_0 -> 0 ─────■─────┤ Rz(π/4) ├───■───────────────────
    «                  ┌─┴─┐   ├─────────┴┐┌─┴─┐            ┌───┐
    «     q0_1 -> 1 ───┤ X ├───┤ Rz(-π/4) ├┤ X ├────────────┤ X ├
    «               ┌──┴───┴──┐├─────────┬┘├───┴┐┌─────────┐└─┬─┘
    «     q0_2 -> 2 ┤ Rz(π/4) ├┤ Rz(π/2) ├─┤ √X ├┤ Rz(π/2) ├──■──
    «               └─────────┘└─────────┘ └────┘└─────────┘     
    «ancilla_0 -> 3 ─────────────────────────────────────────────
    «                                                            
    «ancilla_1 -> 4 ─────────────────────────────────────────────
    «                                                            


When :attr:`~qiskit.transpile.optimization_level` is equals to 1, is a reduction in the cnot
gate and changes in the qubits position. And its connection is
**[[0,1],[1,0],[2,1]]**

.. testcode::

    qc_b1 = transpile(qc,backend = backend,optimization_level = 1)
    
    print(qc_b1)


.. testoutput::

    global phase: 5π/8
                   ┌───┐                                                  »
         q0_1 -> 0 ┤ X ├──────────────────────────────■───────────────────»
                   └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌───┐»
         q0_2 -> 1 ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├»
                        └─────────┘└────┘└─────────┘└───┘└──────────┘└─┬─┘»
         q0_0 -> 2 ────────────────────────────────────────────────────■──»
                                                                          »
    ancilla_0 -> 3 ───────────────────────────────────────────────────────»
                                                                          »
    ancilla_1 -> 4 ───────────────────────────────────────────────────────»
                                                                          »
    «                               ┌─────────┐                                   »
    «     q0_1 -> 0 ─────────────■──┤ Rz(π/4) ├───────────────────────────────────»
    «               ┌─────────┐┌─┴─┐├─────────┴┐┌───┐┌──────────┐┌────┐┌─────────┐»
    «     q0_2 -> 1 ┤ Rz(π/4) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├┤ √X ├┤ Rz(π/2) ├»
    «               └─────────┘└───┘└──────────┘└─┬─┘└──────────┘└────┘└─────────┘»
    «     q0_0 -> 2 ──────────────────────────────■───────────────────────────────»
    «                                                                             »
    «ancilla_0 -> 3 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «ancilla_1 -> 4 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «                    ┌───┐                                
    «     q0_1 -> 0 ──■──┤ X ├──■──────────────────────────■──
    «               ┌─┴─┐└─┬─┘┌─┴─┐┌───┐┌──────────┐┌───┐┌─┴─┐
    «     q0_2 -> 1 ┤ X ├──■──┤ X ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ X ├
    «               └───┘     └───┘└─┬─┘├─────────┬┘└─┬─┘└───┘
    «     q0_0 -> 2 ─────────────────■──┤ Rz(π/4) ├───■───────
    «                                   └─────────┘           
    «ancilla_0 -> 3 ──────────────────────────────────────────
    «                                                         
    «ancilla_1 -> 4 ──────────────────────────────────────────
    «                                                         


When :attr:`~qiskit.transpile.optimization_level` is equals to 2, in a small quantum circuit
sometimes is the same as the light optimization.

.. testcode::

    qc_b2 = transpile(qc,backend = backend,optimization_level = 2)
    
    print(qc_b2)


.. testoutput::

    global phase: 5π/8
                   ┌───┐                                                  »
         q0_1 -> 0 ┤ X ├──────────────────────────────■───────────────────»
                   └─┬─┘┌─────────┐┌────┐┌─────────┐┌─┴─┐┌──────────┐┌───┐»
         q0_2 -> 1 ──■──┤ Rz(π/2) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(-π/4) ├┤ X ├»
                        └─────────┘└────┘└─────────┘└───┘└──────────┘└─┬─┘»
         q0_0 -> 2 ────────────────────────────────────────────────────■──»
                                                                          »
    ancilla_0 -> 3 ───────────────────────────────────────────────────────»
                                                                          »
    ancilla_1 -> 4 ───────────────────────────────────────────────────────»
                                                                          »
    «                               ┌─────────┐                                   »
    «     q0_1 -> 0 ─────────────■──┤ Rz(π/4) ├───────────────────────────────────»
    «               ┌─────────┐┌─┴─┐├─────────┴┐┌───┐┌──────────┐┌────┐┌─────────┐»
    «     q0_2 -> 1 ┤ Rz(π/4) ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├┤ √X ├┤ Rz(π/2) ├»
    «               └─────────┘└───┘└──────────┘└─┬─┘└──────────┘└────┘└─────────┘»
    «     q0_0 -> 2 ──────────────────────────────■───────────────────────────────»
    «                                                                             »
    «ancilla_0 -> 3 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «ancilla_1 -> 4 ──────────────────────────────────────────────────────────────»
    «                                                                             »
    «                    ┌───┐                                
    «     q0_1 -> 0 ──■──┤ X ├──■──────────────────────────■──
    «               ┌─┴─┐└─┬─┘┌─┴─┐┌───┐┌──────────┐┌───┐┌─┴─┐
    «     q0_2 -> 1 ┤ X ├──■──┤ X ├┤ X ├┤ Rz(-π/4) ├┤ X ├┤ X ├
    «               └───┘     └───┘└─┬─┘├─────────┬┘└─┬─┘└───┘
    «     q0_0 -> 2 ─────────────────■──┤ Rz(π/4) ├───■───────
    «                                   └─────────┘           
    «ancilla_0 -> 3 ──────────────────────────────────────────
    «                                                         
    «ancilla_1 -> 4 ──────────────────────────────────────────
    «                                                         


When :attr:`~qiskit.transpile.optimization_level` is equals to 3,is the combination of the
previous parameters in try to reduce the gates, find the best coupling
map connection **[[0,1],[1,0],[2,1]]**.

.. testcode::

    qc_b3 = transpile(qc,backend = backend,optimization_level = 3)
    
    print(qc_b3)


.. testoutput::

    global phase: 3π/8
                      ┌────────┐  ┌────┐ ┌────────┐        ┌────┐        »
         q0_1 -> 0 ───┤ Rz(-π) ├──┤ √X ├─┤ Rz(-π) ├──■─────┤ √X ├────────»
                   ┌──┴────────┴─┐├────┤┌┴────────┤┌─┴─┐┌──┴────┴─┐┌────┐»
         q0_2 -> 1 ┤ Rz(-2.3821) ├┤ √X ├┤ Rz(π/2) ├┤ X ├┤ Rz(π/2) ├┤ √X ├»
                   └─────────────┘└────┘└─────────┘└───┘└─────────┘└────┘»
         q0_0 -> 2 ──────────────────────────────────────────────────────»
                                                                         »
    ancilla_0 -> 3 ──────────────────────────────────────────────────────»
                                                                         »
    ancilla_1 -> 4 ──────────────────────────────────────────────────────»
                                                                         »
    «                                                                     »
    «     q0_1 -> 0 ───────────────────────────────────────────────────■──»
    «               ┌──────────────┐┌────┐┌─────────┐┌───┐┌─────────┐┌─┴─┐»
    «     q0_2 -> 1 ┤ Rz(-0.75949) ├┤ √X ├┤ Rz(π/4) ├┤ X ├┤ Rz(π/4) ├┤ X ├»
    «               └──────────────┘└────┘└─────────┘└─┬─┘└─────────┘└───┘»
    «     q0_0 -> 2 ───────────────────────────────────■──────────────────»
    «                                                                     »
    «ancilla_0 -> 3 ──────────────────────────────────────────────────────»
    «                                                                     »
    «ancilla_1 -> 4 ──────────────────────────────────────────────────────»
    «                                                                     »
    «               ┌─────────┐                                        ┌───┐     »
    «     q0_1 -> 0 ┤ Rz(π/4) ├─────────────────────────────────────■──┤ X ├──■──»
    «               ├─────────┴┐┌───┐┌──────────┐┌────┐┌─────────┐┌─┴─┐└─┬─┘┌─┴─┐»
    «     q0_2 -> 1 ┤ Rz(-π/4) ├┤ X ├┤ Rz(3π/4) ├┤ √X ├┤ Rz(π/2) ├┤ X ├──■──┤ X ├»
    «               └──────────┘└─┬─┘└──────────┘└────┘└─────────┘└───┘     └───┘»
    «     q0_0 -> 2 ──────────────■──────────────────────────────────────────────»
    «                                                                            »
    «ancilla_0 -> 3 ─────────────────────────────────────────────────────────────»
    «                                                                            »
    «ancilla_1 -> 4 ─────────────────────────────────────────────────────────────»
    «                                                                            »
    «                                          
    «     q0_1 -> 0 ────────────────────────■──
    «               ┌───┐┌──────────┐┌───┐┌─┴─┐
    «     q0_2 -> 1 ┤ X ├┤ Rz(-π/4) ├┤ X ├┤ X ├
    «               └─┬─┘├─────────┬┘└─┬─┘└───┘
    «     q0_0 -> 2 ──■──┤ Rz(π/4) ├───■───────
    «                    └─────────┘           
    «ancilla_0 -> 3 ───────────────────────────
    «                                          
    «ancilla_1 -> 4 ───────────────────────────
    «                                          


Plotting the Results
====================

In this section, you can visualize the results of the previous examples by considering the depth,
 the number of gates, and the number of cx gates in different plots for the previous quantum circuits.

.. testcode::

    import matplotlib.pyplot as plt
    
    
    
    fig, ax = plt.subplots()
    my_xticks = [str(i) for i in range(4)]
    plt.xticks(range(4), my_xticks)
    ax.plot(range(4), [qc_bg0.depth(),qc_bg1.depth(),qc_bg2.depth(),qc_bg3.depth()],label = "basis_gates parameter", marker='o',color ='#6929C4')
    ax.plot(range(4), [qc_il0.depth(),qc_il1.depth(),qc_il2.depth(),qc_il3.depth()],label = "init_layout parameter", marker='o',color ='blue')
    ax.plot(range(4), [qc_cm0.depth(),qc_cm1.depth(),qc_cm2.depth(),qc_cm3.depth()],label = "coupling_map parameter", marker='o',color ='green')
    ax.plot(range(4), [qc_b0.depth(),qc_b1.depth(),qc_b2.depth(),qc_b3.depth()],label = "backend parameter", marker='o',color ='red')
    
    ax.set_title('Results of the depth when applying different optimization levels.')
    ax.set_xlabel('Optimization Level')
    ax.set_ylabel('Quantum Circuits depth')
    plt.legend(bbox_to_anchor =(0.85, 1.))
    
    
    
    fig, ax = plt.subplots()
    my_xticks = [str(i) for i in range(4)]
    plt.xticks(range(4), my_xticks)
    ax.plot(range(4), [qc_bg0.size(),qc_bg1.size(),qc_bg2.size(),qc_bg3.size()],label = "basis_gates parameter", marker='o',color ='#6929C4')
    ax.plot(range(4), [qc_il0.size(),qc_il1.size(),qc_il2.size(),qc_il3.size()],label = "init_layout parameter", marker='o',color ='blue')
    ax.plot(range(4), [qc_cm0.size(),qc_cm1.size(),qc_cm2.size(),qc_cm3.size()],label = "coupling_map parameter", marker='o',color ='green')
    ax.plot(range(4), [qc_b0.size(),qc_b1.size(),qc_b2.size(),qc_b3.size()],label = "backend parameter", marker='o',color ='red')
    
    ax.set_title('Results of the number of gates when applying different optimization levels.')
    ax.set_xlabel('Optimization Level')
    ax.set_ylabel('Number of gates')
    
    
    
    fig, ax = plt.subplots()
    my_xticks = [str(i) for i in range(4)]
    plt.xticks(range(4), my_xticks)
    ax.plot(range(4), [qc_bg0.num_nonlocal_gates(),qc_bg1.num_nonlocal_gates(),qc_bg2.num_nonlocal_gates(),qc_bg3.num_nonlocal_gates()],label = "basis_gates parameter", marker='o',color ='#6929C4')
    ax.plot(range(4), [qc_il0.num_nonlocal_gates(),qc_il1.num_nonlocal_gates(),qc_il2.num_nonlocal_gates(),qc_il3.num_nonlocal_gates()],label = "init_layout parameter", marker='o',color ='blue')
    ax.plot(range(4), [qc_cm0.num_nonlocal_gates(),qc_cm1.num_nonlocal_gates(),qc_cm2.num_nonlocal_gates(),qc_cm3.num_nonlocal_gates()],label = "coupling_map parameter", marker='o',color ='green')
    ax.plot(range(4), [qc_b0.num_nonlocal_gates(),qc_b1.num_nonlocal_gates(),qc_b2.num_nonlocal_gates(),qc_b3.num_nonlocal_gates()],label = "backend parameter", marker='o',color ='red')
    
    ax.set_title('Results of cx gates when applying different optimization levels.')
    ax.set_xlabel('Optimization Level')
    ax.set_ylabel('Number of cx gates')
    





.. testoutput::

    Text(0, 0.5, 'Number of cx gates')




.. image:: ../source_images/depth.png



.. image:: ../source_images/num_gates.png



.. image:: ../source_images/num_cx_gates.png

