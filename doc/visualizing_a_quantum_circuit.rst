


Visualizing a Quantum Circuit
=============================

.. code:: ipython3

    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

Drawing a Quantum Circuit
-------------------------

When building a quantum circuit it often helps to draw the circuit. This
is supported natively by a ``QuantumCircuit`` object. You can either
just call ``print()`` on the circuit or call the ``draw()`` method on
the object. This will render a `ASCII art
version <https://en.wikipedia.org/wiki/ASCII_art>`__ of the circuit
diagram.

.. code:: ipython3

    # Build a quantum circuit
    
    n = 3  # number of qubits 
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    
    circuit = QuantumCircuit(q, c)
    
    circuit.x(q[1])
    circuit.h(q)
    circuit.cx(q[0], q[1])
    circuit.measure(q, c);

.. code:: ipython3

    print(circuit)


.. parsed-literal::

                               ┌───┐        ┌─┐
    q0_0: |0>──────────────────┤ H ├──■─────┤M├
                     ┌───┐┌───┐└───┘┌─┴─┐┌─┐└╥┘
    q0_1: |0>────────┤ X ├┤ H ├─────┤ X ├┤M├─╫─
             ┌───┐┌─┐└───┘└───┘     └───┘└╥┘ ║ 
    q0_2: |0>┤ H ├┤M├─────────────────────╫──╫─
             └───┘└╥┘                     ║  ║ 
     c0_0: 0 ══════╬══════════════════════╬══╩═
                   ║                      ║    
     c0_1: 0 ══════╬══════════════════════╩════
                   ║                           
     c0_2: 0 ══════╩═══════════════════════════
                                               


.. code:: ipython3

    circuit.draw()




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">                           ┌───┐        ┌─┐
    q0_0: |0>──────────────────┤ H ├──■─────┤M├
                     ┌───┐┌───┐└───┘┌─┴─┐┌─┐└╥┘
    q0_1: |0>────────┤ X ├┤ H ├─────┤ X ├┤M├─╫─
             ┌───┐┌─┐└───┘└───┘     └───┘└╥┘ ║ 
    q0_2: |0>┤ H ├┤M├─────────────────────╫──╫─
             └───┘└╥┘                     ║  ║ 
     c0_0: 0 ══════╬══════════════════════╬══╩═
                   ║                      ║    
     c0_1: 0 ══════╬══════════════════════╩════
                   ║                           
     c0_2: 0 ══════╩═══════════════════════════
                                               </pre>



Alternative Renderers for Circuits
----------------------------------

While a text output is useful for quickly seeing the output while
developing a circuit it doesn’t provide the most flexibility in it’s
output. There are 2 other alternative output renderers for the quantum
circuit. One uses `matplotlib <https://matplotlib.org/>`__ and the other
uses `LaTex <https://www.latex-project.org/>`__ leveraging the `qcircuit
package <https://github.com/CQuIC/qcircuit>`__. These can be specified
by using ``mpl`` and ``latex`` values for the ``output`` kwarg on the
draw() method.

.. code:: ipython3

    # Matplotlib Drawing
    circuit.draw(output='mpl')




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_8_0.png



.. code:: ipython3

    # Latex Drawing
    circuit.draw(output='latex')




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_9_0.png



Controlling output from circuit.draw()
--------------------------------------

By default the draw method returns the rendered image as an object and
does not output anything. The exact class returned depends on the output
specified: ``'text'``\ (the default returns a ``TextDrawer`` object,
``'mpl'`` returns a ``matplotlib.Figure`` object, and ``latex`` returns
a ``PIL.Image`` object. Having the return types enables modifying or
directly interacting with the rendered output from the drawers. Jupyter
notebooks understand these return types and render it for us in this
tutorial, but when running outside of jupyter you do not have this
feature automatically. However, the ``draw()`` method has optional
arguments to display or save the output. When specified the ``filename``
kwarg takes a path to save the rendered output to. Or if you’re using
the ``mpl`` or ``latex`` outputs you can leverage the ``interactive``
kwarg to open the image in a new window (this will not always work from
within a notebook but will be demonstrated anyway).

Customizing the output
----------------------

Depending on the output there are also options to customize the circuit
diagram rendered by the circuit.

Disable Plot Barriers and Reversing Bit Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first two options are shared between all 3 backends and they let you
configure both the bit orders and whether or not you draw barriers.
These can be set by the ``reverse_bits`` kwarg and ``plot_barriers``
kwarg respectively. The examples below will work with any output
backend, only ``latex`` is used for brevity.

.. code:: ipython3

    # Draw a new circuit with barriers and more registers
    
    q_a = QuantumRegister(3, name='qa')
    q_b = QuantumRegister(5, name='qb')
    c_a = ClassicalRegister(3)
    c_b = ClassicalRegister(5)
    
    circuit = QuantumCircuit(q_a, q_b, c_a, c_b)
    
    circuit.x(q_a[1])
    circuit.x(q_b[1])
    circuit.x(q_b[2])
    circuit.x(q_b[4])
    circuit.barrier()
    circuit.h(q_a)
    circuit.barrier(q_a)
    circuit.h(q_b)
    circuit.cswap(q_b[0], q_b[1], q_b[2])
    circuit.cswap(q_b[2], q_b[3], q_b[4])
    circuit.cswap(q_b[3], q_b[4], q_b[0])
    circuit.barrier(q_b)
    circuit.measure(q_a, c_a)
    circuit.measure(q_b, c_b);

.. code:: ipython3

    # Draw the circuit
    circuit.draw(output='latex')




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_13_0.png



.. code:: ipython3

    # Draw the circuit with reversed bit order
    circuit.draw(output='latex', reverse_bits=True)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_14_0.png



.. code:: ipython3

    # Draw the circuit without barriers
    circuit.draw(output='latex', plot_barriers=False)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_15_0.png



.. code:: ipython3

    # Draw the circuit without barriers and reverse bit order
    circuit.draw(output='latex', plot_barriers=False, reverse_bits=True)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_16_0.png



Backend specific customizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are also some options available to customize the output diagram
which only work for a specific backend. The ``line_length`` kwarg for
the ``text`` backend which can be used to set a maximum width for the
output. When a diagram is wider than that it will wrap the diagram
below. The ``mpl`` backend has the ``style`` kwarg which is used to
customize the output. The ``scale`` option is used by the ``mpl`` and
``latex`` backends to adjust the size of the output image, it’s a
multiplicative adjustment factor used to scale the output size. The
``style`` kwarg takes in a dict with many different options in it. It
provides a high level of flexibility and enables things like changing
colors, changing rendered text for different types of gates, different
line styles, etc. The list of available options for this are:

-  **textcolor** (str): The color code to use for text. Defaults to
   ``'#000000'``
-  **subtextcolor** (str): The color code to use for subtext. Defaults
   to ``'#000000'``
-  **linecolor** (str): The color code to use for lines. Defaults to
   ``'#000000'``
-  **creglinecolor** (str): The color code to use for classical register
   lines ``'#778899'``
-  **gatetextcolor** (str): The color code to use for gate text
   ``'#000000'``
-  **gatefacecolor** (str): The color code to use for gates. Defaults to
   ``'#ffffff'``
-  **barrierfacecolor** (str): The color code to use for barriers.
   Defaults to ``'#bdbdbd'``
-  **backgroundcolor** (str): The color code to use for the background.
   Defaults to ``'#ffffff'``
-  **fontsize** (int): The font size to use for text. Defaults to 13
-  **subfontsize** (int): The font size to use for subtext. Defaults to
   8
-  **displaytext** (dict): A dictionary of the text to use for each
   element type in the output visualization. The default values are:

   { ‘id’: ‘id’, ‘u0’: ‘U_0’, ‘u1’: ‘U_1’, ‘u2’: ‘U_2’, ‘u3’: ‘U_3’,
   ‘x’: ‘X’, ‘y’: ‘Y’, ‘z’: ‘Z’, ‘h’: ‘H’, ‘s’: ‘S’, ‘sdg’: ‘S^\dagger’,
   ‘t’: ‘T’, ‘tdg’: ‘T^\dagger’, ‘rx’: ‘R_x’, ‘ry’: ‘R_y’, ‘rz’: ‘R_z’,
   ‘reset’: ‘\\left|0\right\rangle’ }

   You must specify all the necessary values if using this. There is no
   provision for passing an incomplete dict in.
-  **displaycolor** (dict): The color codes to use for each circuit
   element. By default all values default to the value of
   ``gatefacecolor`` and the keys are the same as ``displaytext``. Also,
   just like ``displaytext`` there is no provision for an incomplete
   dict passed in.
-  **latexdrawerstyle** (bool): When set to True enable latex mode which
   will draw gates like the ``latex`` output modes.
-  **usepiformat** (bool): When set to True use radians for output
-  **fold** (int): The number of circuit elements to fold the circuit
   at. Defaults to 20
-  **cregbundle** (bool): If set True bundle classical registers
-  **showindex** (bool): If set True draw an index.
-  **compress** (bool): If set True draw a compressed circuit
-  **figwidth** (int): The maximum width (in inches) for the output
   figure.
-  **dpi** (int): The DPI to use for the output image. Defaults to 150
-  **creglinestyle** (str): The style of line to use for classical
   registers. Choices are ``'solid'``, ``'doublet'``, or any valid
   matplotlib ``linestyle`` kwarg value. Defaults to ``doublet``

.. code:: ipython3

    # Set line length to 80 for above circuit
    circuit.draw(output='text', line_length=80)




.. raw:: html

    <pre style="word-wrap: normal;white-space: pre;line-height: 15px;">                              ░                                               »
    qa_0: |0>─────────────────────░───────────────────────────────────────────────»
                            ┌───┐ ░                                               »
    qa_1: |0>───────────────┤ X ├─░───────────────────────────────────────────────»
                            └───┘ ░                                               »
    qa_2: |0>─────────────────────░───────────────────────────────────────────────»
                                  ░                     ┌───┐          ░          »
    qb_0: |0>─────────────────────░─────────────────────┤ H ├─■─────X──░──────────»
                       ┌───┐      ░                ┌───┐└───┘ │     │  ░          »
    qb_1: |0>──────────┤ X ├──────░────────────────┤ H ├──────X─────┼──░──────────»
                  ┌───┐└───┘      ░           ┌───┐└───┘      │     │  ░       ┌─┐»
    qb_2: |0>─────┤ X ├───────────░───────────┤ H ├───────────X──■──┼──░───────┤M├»
                  └───┘           ░      ┌───┐└───┘              │  │  ░    ┌─┐└╥┘»
    qb_3: |0>─────────────────────░──────┤ H ├───────────────────X──■──░────┤M├─╫─»
             ┌───┐                ░ ┌───┐└───┘                   │  │  ░ ┌─┐└╥┘ ║ »
    qb_4: |0>┤ X ├────────────────░─┤ H ├────────────────────────X──X──░─┤M├─╫──╫─»
             └───┘                ░ └───┘                              ░ └╥┘ ║  ║ »
     c1_0: 0 ═════════════════════════════════════════════════════════════╬══╬══╬═»
                                                                          ║  ║  ║ »
     c1_1: 0 ═════════════════════════════════════════════════════════════╬══╬══╬═»
                                                                          ║  ║  ║ »
     c1_2: 0 ═════════════════════════════════════════════════════════════╬══╬══╬═»
                                                                          ║  ║  ║ »
     c2_0: 0 ═════════════════════════════════════════════════════════════╬══╬══╬═»
                                                                          ║  ║  ║ »
     c2_1: 0 ═════════════════════════════════════════════════════════════╬══╬══╬═»
                                                                          ║  ║  ║ »
     c2_2: 0 ═════════════════════════════════════════════════════════════╬══╬══╩═»
                                                                          ║  ║    »
     c2_3: 0 ═════════════════════════════════════════════════════════════╬══╩════»
                                                                          ║       »
     c2_4: 0 ═════════════════════════════════════════════════════════════╩═══════»
                                                                                  »
    «                      ┌───┐ ░       ┌─┐
    «qa_0: ────────────────┤ H ├─░───────┤M├
    «                 ┌───┐└───┘ ░    ┌─┐└╥┘
    «qa_1: ───────────┤ H ├──────░────┤M├─╫─
    «            ┌───┐└───┘      ░ ┌─┐└╥┘ ║ 
    «qa_2: ──────┤ H ├───────────░─┤M├─╫──╫─
    «         ┌─┐└───┘           ░ └╥┘ ║  ║ 
    «qb_0: ───┤M├───────────────────╫──╫──╫─
    «      ┌─┐└╥┘                   ║  ║  ║ 
    «qb_1: ┤M├─╫────────────────────╫──╫──╫─
    «      └╥┘ ║                    ║  ║  ║ 
    «qb_2: ─╫──╫────────────────────╫──╫──╫─
    «       ║  ║                    ║  ║  ║ 
    «qb_3: ─╫──╫────────────────────╫──╫──╫─
    «       ║  ║                    ║  ║  ║ 
    «qb_4: ─╫──╫────────────────────╫──╫──╫─
    «       ║  ║                    ║  ║  ║ 
    «c1_0: ═╬══╬════════════════════╬══╬══╩═
    «       ║  ║                    ║  ║    
    «c1_1: ═╬══╬════════════════════╬══╩════
    «       ║  ║                    ║       
    «c1_2: ═╬══╬════════════════════╩═══════
    «       ║  ║                            
    «c2_0: ═╬══╩════════════════════════════
    «       ║                               
    «c2_1: ═╩═══════════════════════════════
    «                                       
    «c2_2: ═════════════════════════════════
    «                                       
    «c2_3: ═════════════════════════════════
    «                                       
    «c2_4: ═════════════════════════════════
    «                                       </pre>



.. code:: ipython3

    # Change the background color in mpl
    
    style = {'backgroundcolor': 'lightgreen'}
    
    circuit.draw(output='mpl', style=style)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_19_0.png



.. code:: ipython3

    # Scale the mpl output to 1/2 the normal size
    circuit.draw(output='mpl', scale=0.5)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_20_0.png



.. code:: ipython3

    # Scale the latex output to 1/2 the normal size
    circuit.draw(output='latex', scale=0.5)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_21_0.png



Latex Source
------------

One additional option available with the latex output type is to return
the raw LaTex source code instead of rendering an image for it. This
enables easy integration in a seperate LaTex document. To use this you
can just set the ``output`` kwarg to ``'latex_source'``. You can also
use the ``filename`` kwarg to write this output directly to a file (and
still return the string) instead of returning just a string.

.. code:: ipython3

    # Print the latex source for the visualization
    print(circuit.draw(output='latex_source'))


.. parsed-literal::

    % \documentclass[preview]{standalone}
    % If the image is too large to fit on this documentclass use
    \documentclass[draft]{beamer}
    % img_width = 16, img_depth = 17
    \usepackage[size=custom,height=24,width=28,scale=0.7]{beamerposter}
    % instead and customize the height and width (in cm) to fit.
    % Large images may run out of memory quickly.
    % To fix this use the LuaLaTeX compiler, which dynamically
    % allocates memory.
    \usepackage[braket, qm]{qcircuit}
    \usepackage{amsmath}
    \pdfmapfile{+sansmathaccent.map}
    % \usepackage[landscape]{geometry}
    % Comment out the above line if using the beamer documentclass.
    \begin{document}
    \begin{equation*}
        \Qcircuit @C=0.5em @R=0.0em @!R {
    	 	\lstick{qa_{0}: \ket{0}} & \qw & \qw \barrier{7} & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \gate{H} & \qw \barrier[-1.15em]{2} & \qw & \qw & \meter & \qw & \qw\\
    	 	\lstick{qa_{1}: \ket{0}} & \gate{X} & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \gate{H} & \qw & \qw & \meter & \qw & \qw & \qw\\
    	 	\lstick{qa_{2}: \ket{0}} & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \gate{H} & \qw & \meter & \qw & \qw & \qw & \qw\\
    	 	\lstick{qb_{0}: \ket{0}} & \qw & \qw & \gate{H} & \ctrl{1} & \qw & \qswap \qwx[4] & \qw \barrier[-1.15em]{4} & \qw & \qw & \qw & \meter & \qw & \qw & \qw & \qw & \qw & \qw\\
    	 	\lstick{qb_{1}: \ket{0}} & \gate{X} & \qw & \gate{H} & \qswap & \qw & \qw & \qw & \qw & \qw & \meter & \qw & \qw & \qw & \qw & \qw & \qw & \qw\\
    	 	\lstick{qb_{2}: \ket{0}} & \gate{X} & \qw & \gate{H} & \qswap \qwx[-1] & \ctrl{1} & \qw & \qw & \qw & \meter & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw\\
    	 	\lstick{qb_{3}: \ket{0}} & \qw & \qw & \gate{H} & \qw & \qswap & \ctrl{1} & \qw & \meter & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw\\
    	 	\lstick{qb_{4}: \ket{0}} & \gate{X} & \qw & \gate{H} & \qw & \qswap \qwx[-1] & \qswap & \meter & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw\\
    	 	\lstick{c1_{0}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw\\
    	 	\lstick{c1_{1}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw\\
    	 	\lstick{c1_{2}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw\\
    	 	\lstick{c2_{0}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw & \cw & \cw\\
    	 	\lstick{c2_{1}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw & \cw & \cw & \cw\\
    	 	\lstick{c2_{2}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw\\
    	 	\lstick{c2_{3}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw\\
    	 	\lstick{c2_{4}: 0} & \cw & \cw & \cw & \cw & \cw & \cw & \cw \cwx[-8] & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw & \cw\\
    	 }
    \end{equation*}
    
    \end{document}


.. code:: ipython3

    # Save the latex source to a file
    circuit.draw(output='latex_source', filename='/tmp/circuit.tex');

circuit_drawer() as function
----------------------------

If you have an application where you prefer to draw a circuit with a
self contained function instead of as a method of a circuit object you
can directly use the ``circuit_drawer()`` function, which is part of the
public stable interface from ``qiskit.tools.visualization``. The
function behaves identically to the ``circuit.draw()`` method except
that it takes in a circuit object as required argument.

.. raw:: html

   <div class="alert alert-block alert-info">

Note: In Qiskit Terra <= 0.7 the default behavior for the
circuit_drawer() function is to use the latex output backend, and in
0.6.x that includes a fallback to mpl if latex fails for any reason. But
starting in releases > 0.7 the default changes to use the text output.

.. raw:: html

   </div>

.. code:: ipython3

    from qiskit.tools.visualization import circuit_drawer

.. code:: ipython3

    circuit_drawer(circuit, output='mpl', plot_barriers=False)




.. image:: visualizing_a_quantum_circuit_files/visualizing_a_quantum_circuit_27_0.png


