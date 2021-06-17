# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for monitoring backends."""

import time
import threading
import types
from IPython.display import display
from IPython.core.magic import line_magic, Magics, magics_class
from IPython.core import magic_arguments
import matplotlib.pyplot as plt
import ipywidgets as widgets
from qiskit.tools.monitor.overview import get_unique_backends
from qiskit.visualization.gate_map import plot_gate_map


@magics_class
class BackendOverview(Magics):
    """A class of status magic functions."""

    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        "-i", "--interval", type=float, default=60, help="Interval for status check."
    )
    def qiskit_backend_overview(self, line=""):
        """A Jupyter magic function to monitor backends."""
        args = magic_arguments.parse_argstring(self.qiskit_backend_overview, line)

        unique_hardware_backends = get_unique_backends()
        _value = "<h2 style ='color:#ffffff; background-color:#000000;"
        _value += "padding-top: 1%; padding-bottom: 1%;padding-left: 1%;"
        _value += "margin-top: 0px'>Backend Overview</h2>"
        backend_title = widgets.HTML(value=_value, layout=widgets.Layout(margin="0px 0px 0px 0px"))

        build_back_widgets = [backend_widget(b) for b in unique_hardware_backends]

        _backends = []
        # Sort backends by operational or not
        oper_ord_backends = []
        for n, back in enumerate(unique_hardware_backends):
            if back.status().operational:
                oper_ord_backends = [build_back_widgets[n]] + oper_ord_backends
                _backends = [back] + _backends
            else:
                oper_ord_backends = oper_ord_backends + [build_back_widgets[n]]
                _backends = _backends + [back]

        qubit_label = widgets.Label(value="Num. Qubits")
        qv_label = widgets.Label(value="Quantum Vol.")
        pend_label = widgets.Label(
            value="Pending Jobs", layout=widgets.Layout(margin="5px 0px 0px 0px")
        )
        least_label = widgets.Label(
            value="Least Busy", layout=widgets.Layout(margin="10px 0px 0px 0px")
        )
        oper_label = widgets.Label(
            value="Operational", layout=widgets.Layout(margin="5px 0px 0px 0px")
        )
        t12_label = widgets.Label(
            value="Avg. T1 / T2", layout=widgets.Layout(margin="10px 0px 0px 0px")
        )
        cx_label = widgets.Label(
            value="Avg. CX Err.", layout=widgets.Layout(margin="8px 0px 0px 0px")
        )
        meas_label = widgets.Label(
            value="Avg. Meas. Err.", layout=widgets.Layout(margin="8px 0px 0px 0px")
        )

        labels_widget = widgets.VBox(
            [
                qubit_label,
                qv_label,
                pend_label,
                oper_label,
                least_label,
                t12_label,
                cx_label,
                meas_label,
            ],
            layout=widgets.Layout(margin="295px 0px 0px 0px", min_width="100px"),
        )

        backend_grid = GridBox_with_thread(
            children=oper_ord_backends,
            layout=widgets.Layout(
                grid_template_columns="250px " * len(unique_hardware_backends),
                grid_template_rows="auto",
                grid_gap="0px 25px",
            ),
        )

        backend_grid._backends = _backends  # pylint: disable=attribute-defined-outside-init
        backend_grid._update = types.MethodType(  # pylint: disable=attribute-defined-outside-init
            update_backend_info, backend_grid
        )

        backend_grid._thread = threading.Thread(  # pylint: disable=attribute-defined-outside-init
            target=backend_grid._update, args=(args.interval,)
        )
        backend_grid._thread.start()

        back_box = widgets.HBox([labels_widget, backend_grid])

        back_monitor = widgets.VBox([backend_title, back_box])
        display(back_monitor)


class GridBox_with_thread(widgets.GridBox):  # pylint: disable=invalid-name
    """A GridBox that will close an attached thread"""

    def __del__(self):
        """Object disposal"""
        if hasattr(self, "_thread"):
            try:
                self._thread.do_run = False
                self._thread.join()
            except Exception:  # pylint: disable=broad-except
                pass
        self.close()


def backend_widget(backend):
    """Creates a backend widget."""
    config = backend.configuration().to_dict()
    props = backend.properties().to_dict()

    name = widgets.HTML(value=f"<h4>{backend.name()}</h4>", layout=widgets.Layout())

    num_qubits = config["n_qubits"]

    qv_val = "-"
    if "quantum_volume" in config.keys():
        if config["quantum_volume"]:
            qv_val = config["quantum_volume"]

    qubit_count = widgets.HTML(
        value=f"<h5><b>{num_qubits}</b></h5>",
        layout=widgets.Layout(justify_content="center"),
    )

    qv_value = widgets.HTML(
        value=f"<h5>{qv_val}</h5>",
        layout=widgets.Layout(justify_content="center"),
    )

    cmap = widgets.Output(
        layout=widgets.Layout(
            min_width="250px",
            max_width="250px",
            max_height="250px",
            min_height="250px",
            justify_content="center",
            align_items="center",
            margin="0px 0px 0px 0px",
        )
    )

    with cmap:
        _cmap_fig = plot_gate_map(backend, plot_directed=False, label_qubits=False)
        if _cmap_fig is not None:
            display(_cmap_fig)
            # Prevents plot from showing up twice.
            plt.close(_cmap_fig)

    pending = generate_jobs_pending_widget()

    is_oper = widgets.HTML(value="<h5></h5>", layout=widgets.Layout(justify_content="center"))

    least_busy = widgets.HTML(value="<h5></h5>", layout=widgets.Layout(justify_content="center"))

    t1_units = props["qubits"][0][0]["unit"]
    avg_t1 = round(sum(q[0]["value"] for q in props["qubits"]) / num_qubits, 1)
    avg_t2 = round(sum(q[1]["value"] for q in props["qubits"]) / num_qubits, 1)
    t12_widget = widgets.HTML(
        value=f"<h5>{avg_t1} / {avg_t2} {t1_units}</h5>",
        layout=widgets.Layout(),
    )

    avg_cx_err = "NA"
    if config["coupling_map"]:
        sum_cx_err = 0
        num_cx = 0
        for gate in props["gates"]:
            if gate["gate"] == "cx":
                for param in gate["parameters"]:
                    if param["name"] == "gate_error":
                        # Value == 1.0 means gate effectively off
                        if param["value"] != 1.0:
                            sum_cx_err += param["value"]
                            num_cx += 1
        avg_cx_err = round(sum_cx_err / (num_cx), 4)

    cx_widget = widgets.HTML(value=f"<h5>{avg_cx_err}</h5>", layout=widgets.Layout())

    avg_meas_err = 0
    for qub in props["qubits"]:
        for item in qub:
            if item["name"] == "readout_error":
                avg_meas_err += item["value"]
    avg_meas_err = round(avg_meas_err / num_qubits, 4)
    meas_widget = widgets.HTML(value=f"<h5>{avg_meas_err}</h5>", layout=widgets.Layout())

    out = widgets.VBox(
        [
            name,
            cmap,
            qubit_count,
            qv_value,
            pending,
            is_oper,
            least_busy,
            t12_widget,
            cx_widget,
            meas_widget,
        ],
        layout=widgets.Layout(display="inline-flex", flex_flow="column", align_items="center"),
    )

    out._is_alive = True
    return out


def update_backend_info(self, interval=60):
    """Updates the monitor info
    Called from another thread.
    """
    my_thread = threading.currentThread()
    current_interval = 0
    started = False
    all_dead = False
    stati = [None] * len(self._backends)
    while getattr(my_thread, "do_run", True) and not all_dead:
        if current_interval == interval or started is False:
            for ind, back in enumerate(self._backends):
                _value = self.children[ind].children[2].value
                _head = _value.split("<b>")[0]
                try:
                    _status = back.status()
                    stati[ind] = _status
                except Exception:  # pylint: disable=broad-except
                    self.children[ind].children[2].value = _value.replace(
                        _head, "<h5 style='color:#ff5c49'>"
                    )
                    self.children[ind]._is_alive = False
                else:
                    self.children[ind]._is_alive = True
                    self.children[ind].children[2].value = _value.replace(_head, "<h5>")

            idx = list(range(len(self._backends)))
            pending = [s.pending_jobs for s in stati]
            _, least_idx = zip(*sorted(zip(pending, idx)))

            # Make sure least pending is operational
            for ind in least_idx:
                if stati[ind].operational:
                    least_pending_idx = ind
                    break

            for var in idx:
                if var == least_pending_idx:
                    self.children[var].children[6].value = "<h5 style='color:#34bc6e'>True</h5>"
                else:
                    self.children[var].children[6].value = "<h5 style='color:#dc267f'>False</h5>"

                self.children[var].children[4].children[1].max = max(
                    self.children[var].children[4].children[1].max, pending[var] + 10
                )
                self.children[var].children[4].children[1].value = pending[var]
                if stati[var].operational:
                    self.children[var].children[5].value = "<h5 style='color:#34bc6e'>True</h5>"
                else:
                    self.children[var].children[5].value = "<h5 style='color:#dc267f'>False</h5>"

            started = True
            current_interval = 0
        time.sleep(1)
        all_dead = not any(wid._is_alive for wid in self.children)
        current_interval += 1


def generate_jobs_pending_widget():
    """Generates a jobs_pending progress bar widget."""
    pbar = widgets.IntProgress(
        value=0,
        min=0,
        max=50,
        description="",
        orientation="horizontal",
        layout=widgets.Layout(max_width="180px"),
    )
    pbar.style.bar_color = "#71cddd"

    pbar_current = widgets.Label(value=str(pbar.value), layout=widgets.Layout(min_width="auto"))
    pbar_max = widgets.Label(value=str(pbar.max), layout=widgets.Layout(min_width="auto"))

    def _on_max_change(change):
        pbar_max.value = str(change["new"])

    def _on_val_change(change):
        pbar_current.value = str(change["new"])

    pbar.observe(_on_max_change, names="max")
    pbar.observe(_on_val_change, names="value")

    jobs_widget = widgets.HBox(
        [pbar_current, pbar, pbar_max],
        layout=widgets.Layout(max_width="250px", min_width="250px", justify_content="center"),
    )

    return jobs_widget
