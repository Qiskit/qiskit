qiskit.tools.visualization package
==================================


.. automodule:: qiskit.tools.visualization
    :members: HAS_MATPLOTLIB
    :undoc-members:
    :show-inheritance:
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       
       
       
       
       
       
       
       
       
       
       qiskit.tools.visualization.dag_visualization

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.tools.visualization.interactive

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`VisualizationError <qiskit.tools.visualization._error.VisualizationError>`
         - For visualization specific errors.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`_has_connection <qiskit._util._has_connection>`
         - Checks to see if internet connection exists to host via specified port
       * - :func:`_text_circuit_drawer <qiskit.tools.visualization._circuit_visualization._text_circuit_drawer>`
         - Draws a circuit using ascii art.
       * - :func:`circuit_drawer <qiskit.tools.visualization._circuit_visualization.circuit_drawer>`
         - Draw a quantum circuit to different formats (set by output parameter): 0.
       * - :func:`generate_latex_source <qiskit.tools.visualization._circuit_visualization.generate_latex_source>`
         - Convert QuantumCircuit to LaTeX string.
       * - :func:`latex_circuit_drawer <qiskit.tools.visualization._circuit_visualization.latex_circuit_drawer>`
         - Draw a quantum circuit based on latex (Qcircuit package)
       * - :func:`matplotlib_circuit_drawer <qiskit.tools.visualization._circuit_visualization.matplotlib_circuit_drawer>`
         - Draw a quantum circuit based on matplotlib.
       * - :func:`plot_bloch_multivector <qiskit.tools.visualization._state_visualization.plot_bloch_multivector>`
         - Plot the Bloch sphere.
       * - :func:`plot_bloch_vector <qiskit.tools.visualization._state_visualization.plot_bloch_vector>`
         - Plot the Bloch sphere.
       * - :func:`plot_circuit <qiskit.tools.visualization._circuit_visualization.plot_circuit>`
         - Plot and show circuit (opens new window, cannot inline in Jupyter)
       * - :func:`plot_histogram <qiskit.tools.visualization._counts_visualization.plot_histogram>`
         - Plot a histogram of data.
       * - :func:`plot_state <qiskit.tools.visualization._state_visualization.plot_state>`
         - Plot the quantum state.
       * - :func:`plot_state_city <qiskit.tools.visualization._state_visualization.plot_state_city>`
         - Plot the cityscape of quantum state.
       * - :func:`plot_state_hinton <qiskit.tools.visualization._state_visualization.plot_state_hinton>`
         - Plot a hinton diagram for the quanum state.
       * - :func:`plot_state_paulivec <qiskit.tools.visualization._state_visualization.plot_state_paulivec>`
         - Plot the paulivec representation of a quantum state.
       * - :func:`plot_state_qsphere <qiskit.tools.visualization._state_visualization.plot_state_qsphere>`
         - Plot the qsphere representation of a quantum state.
       * - :func:`qx_color_scheme <qiskit.tools.visualization._circuit_visualization.qx_color_scheme>`
         - Return default style for matplotlib_circuit_drawer (IBM QX style).
    