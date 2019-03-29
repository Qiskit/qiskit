define([
    'base/js/namespace',
    'base/js/events'
], function (Jupyter, events) {

    // Adds a cell above current cell (will be top if no cells)
    var add_cell = function () {
        Jupyter.notebook.
            insert_cell_above('code').
            // Define default cell here
            set_text(`import numpy as np
# Qiskit
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import transpile
from qiskit.tools.jupyter import *
# Visualization
import matplotlib.pyplot as plt
%matplotlib inline`);
        Jupyter.notebook.select_prev();
        Jupyter.notebook.execute_cell_and_select_below();
    };
    // Button to add
    var QiskitButton = function () {
        Jupyter.toolbar.add_buttons_group([
            Jupyter.keyboard_manager.actions.register({
                'help': 'Qiskit functions',
                'icon': "fa-rocket",
                'handler': add_cell
            }, 'add-default-cell', 'Default cell')
        ])
    }
    return {
        load_ipython_extension: QiskitButton
    };
});