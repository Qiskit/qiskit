###########################
Visualize a quantum circuit
###########################

This guide shows you how to visualize a quantum circuit.

Create a circuit
=================

Let's first create a quantum circuit for the visualization.

.. testcode::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))

Visualize the circuit
=====================

You can visualize a quantum circuit in the following ways:

1. Using the ``print()`` function.
2. Using the :meth:`~.QuantumCircuit.draw` method.
3. Using the :func:`~.circuit_drawer` function.

Using the ``print()`` function
------------------------------

If you call the ``print()`` function on a :class:`~.QuantumCircuit` object, you will get an `ASCII art version <https://en.wikipedia.org/wiki/ASCII_art>`_ of the circuit diagram.

.. testcode::

    print(qc)

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐     ┌─┐   
    q_0: ┤ H ├──■──┤M├───
         ├───┤┌─┴─┐└╥┘┌─┐
    q_1: ┤ H ├┤ X ├─╫─┤M├
         ├───┤└┬─┬┘ ║ └╥┘
    q_2: ┤ H ├─┤M├──╫──╫─
         └───┘ └╥┘  ║  ║ 
    c: 3/═══════╩═══╩══╩═
                2   0  1 

Using the :meth:`~.QuantumCircuit.draw` method
----------------------------------------------

You can also use the :meth:`.QuantumCircuit.draw` method to visualize it. The default output style is 'text', which will output an ASCII art version, the same as using the ``print()`` function.

.. code-block:: python

    qc.draw()

.. code-block:: text

         ┌───┐     ┌─┐   
    q_0: ┤ H ├──■──┤M├───
         ├───┤┌─┴─┐└╥┘┌─┐
    q_1: ┤ H ├┤ X ├─╫─┤M├
         ├───┤└┬─┬┘ ║ └╥┘
    q_2: ┤ H ├─┤M├──╫──╫─
         └───┘ └╥┘  ║  ║ 
    c: 3/═══════╩═══╩══╩═
                2   0  1  

You can also change the output style using the ``output`` argument. These are the available output styles:

1. ``'text'``: renders the circuit with ASCII art. It's the default option.
2. ``'mpl'``: uses `matplotlib <https://matplotlib.org/>`_ to render the circuit.
3. ``'latex'``: uses :math:`\LaTeX` to render the circuit. It requires a full `LaTeX <https://latex.org/forum/>`_ distribution and the package ``pdflatex``.
4. ``'latex_source'``: outputs the :math:`\LaTeX` source code that creates the ``'latex'`` rendering of the circuit.

Because this optional keyword argument is the first of this method, you can type ``qc.draw(option)`` instead of ``qc.draw(output=option)``


``'mpl'`` ouptut
^^^^^^^^^^^^^^^^

.. code-block:: python

    qc.draw('mpl')

.. plot::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))
    qc.draw("mpl")


``'latex_source'`` output
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    qc.draw('latex_source')

.. code-block:: text

    '\\documentclass[border=2px]{standalone}\n\n\\usepackage[braket, qm]{qcircuit}\n\\usepackage{graphicx}\n\n\\begin{document}\n\\scalebox{1.0}{\n\\Qcircuit @C=1.0em @R=0.2em @!R { \\\\\n\t \t\\nghost{{q}_{0} :  } & \\lstick{{q}_{0} :  } & \\gate{\\mathrm{H}} & \\ctrl{1} & \\meter & \\qw & \\qw & \\qw\\\\\n\t \t\\nghost{{q}_{1} :  } & \\lstick{{q}_{1} :  } & \\gate{\\mathrm{H}} & \\targ & \\qw & \\meter & \\qw & \\qw\\\\\n\t \t\\nghost{{q}_{2} :  } & \\lstick{{q}_{2} :  } & \\gate{\\mathrm{H}} & \\meter & \\qw & \\qw & \\qw & \\qw\\\\\n\t \t\\nghost{\\mathrm{{c} :  }} & \\lstick{\\mathrm{{c} :  }} & \\lstick{/_{_{3}}} \\cw & \\dstick{_{_{\\hspace{0.0em}2}}} \\cw \\ar @{<=} [-1,0] & \\dstick{_{_{\\hspace{0.0em}0}}} \\cw \\ar @{<=} [-3,0] & \\dstick{_{_{\\hspace{0.0em}1}}} \\cw \\ar @{<=} [-2,0] & \\cw & \\cw\\\\\n\\\\ }}\n\\end{document}'


Using the :func:`~.circuit_drawer` function
-------------------------------------------

If you prefer to use a self-contained function instead of a :class:`~.QuantumCircuit` method to draw your circuit, you can do it with :func:`~.circuit_drawer` from :mod:`qiskit.visualization`. It has the exact same behavior as the :meth:`~.QuantumCircuit.draw` method above, except that it requires the circuit to be included as an argument.

.. code-block:: python

    from qiskit.visualization import circuit_drawer

    circuit_drawer(qc, output='mpl')

.. plot::

    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))
    circuit_drawer(qc, output='mpl')