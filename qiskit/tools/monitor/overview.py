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

""" A module for viewing the details of all available devices.
"""

import math
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError


def get_unique_backends():
    """Gets the unique backends that are available.

    Returns:
        list: Unique available backends.

    Raises:
        QiskitError: No backends available.
        MissingOptionalLibraryError: If qiskit-ibmq-provider is not installed
    """
    try:
        from qiskit.providers.ibmq import IBMQ
    except ImportError as ex:
        raise MissingOptionalLibraryError(
            libname="qiskit-ibmq-provider",
            name="get_unique_backends",
            pip_install="pip install qiskit-ibmq-provider",
        ) from ex
    backends = []
    for provider in IBMQ.providers():
        for backend in provider.backends():
            backends.append(backend)
    unique_hardware_backends = []
    unique_names = []
    for back in backends:
        if back.name() not in unique_names and not back.configuration().simulator:
            unique_hardware_backends.append(back)
            unique_names.append(back.name())
    if not unique_hardware_backends:
        raise QiskitError("No backends available.")
    return unique_hardware_backends


def backend_monitor(backend):
    """Monitor a single IBMQ backend.

    Args:
        backend (IBMQBackend): Backend to monitor.
    Raises:
        QiskitError: Input is not a IBMQ backend.
        MissingOptionalLibraryError: If qiskit-ibmq-provider is not installed

    Examples:
    .. code-block:: python

       from qiskit import IBMQ
       from qiskit.tools.monitor import backend_monitor
       provider = IBMQ.get_provider(hub='ibm-q')
       backend_monitor(provider.backends.ibmq_lima)
    """
    try:
        from qiskit.providers.ibmq import IBMQBackend
    except ImportError as ex:
        raise MissingOptionalLibraryError(
            libname="qiskit-ibmq-provider",
            name="backend_monitor",
            pip_install="pip install qiskit-ibmq-provider",
        ) from ex

    if not isinstance(backend, IBMQBackend):
        raise QiskitError("Input variable is not of type IBMQBackend.")
    config = backend.configuration().to_dict()
    status = backend.status().to_dict()
    config_dict = {**status, **config}

    print(backend.name())
    print("=" * len(backend.name()))
    print("Configuration")
    print("-" * 13)
    offset = "    "

    upper_list = [
        "n_qubits",
        "operational",
        "status_msg",
        "pending_jobs",
        "backend_version",
        "basis_gates",
        "local",
        "simulator",
    ]

    lower_list = list(set(config_dict.keys()).difference(upper_list))
    # Remove gates because they are in a different tab
    lower_list.remove("gates")
    for item in upper_list + lower_list:
        print(offset + item + ":", config_dict[item])

    # Stop here if simulator
    if config["simulator"]:
        return

    print()
    props = backend.properties()
    qubit_header = None
    sep = " / "

    for index, qubit_data in enumerate(props.qubits):
        name = "Q%s" % index
        gate_data = [gate for gate in props.gates if gate.qubits == [index]]

        cal_data = dict.fromkeys(["T1", "T2", "frequency", "readout_error"], "Unknown")
        for nduv in qubit_data:
            if nduv.name in cal_data:
                cal_data[nduv.name] = format(nduv.value, ".5f") + " " + nduv.unit

        gate_names = []
        gate_error = []
        for gd in gate_data:
            if gd.gate in ["id"]:
                continue
            try:
                gate_error.append(format(props.gate_error(gd.gate, index), ".5f"))
                gate_names.append(gd.gate.upper() + " err")
            except QiskitError:
                pass

        if not qubit_header:
            qubit_header = (
                "Qubits [Name / Freq / T1 / T2" + sep.join([""] + gate_names) + " / Readout err]"
            )
            print(qubit_header)
            print("-" * len(qubit_header))

        qstr = sep.join(
            [name, cal_data["frequency"], cal_data["T1"], cal_data["T2"]]
            + gate_error
            + [cal_data["readout_error"]]
        )

        print(offset + qstr)

    print()
    multi_qubit_gates = [g for g in props.gates if len(g.qubits) > 1]
    multi_header = "Multi-Qubit Gates [Name / Type / Gate Error]"
    print(multi_header)
    print("-" * len(multi_header))

    for gate in multi_qubit_gates:
        qubits = gate.qubits
        ttype = gate.gate
        error = "Unknown"
        try:
            error = format(props.gate_error(gate.gate, qubits), ".5f")
        except QiskitError:
            pass
        mstr = sep.join([f"{ttype}{qubits[0]}_{qubits[1]}", ttype, str(error)])
        print(offset + mstr)


def backend_overview():
    """Gives overview information on all the IBMQ
    backends that are available.

    Examples:

        .. code-block:: python

            from qiskit import IBMQ
            from qiskit.tools.monitor import backend_overview
            provider = IBMQ.get_provider(hub='ibm-q')
            backend_overview()
    """
    unique_hardware_backends = get_unique_backends()
    _backends = []
    # Sort backends by operational or not
    for idx, back in enumerate(unique_hardware_backends):
        if back.status().operational:
            _backends = [back] + _backends
        else:
            _backends = _backends + [back]

    stati = [back.status() for back in _backends]
    idx = list(range(len(_backends)))
    pending = [s.pending_jobs for s in stati]
    _, least_idx = zip(*sorted(zip(pending, idx)))

    # Make sure least pending is operational
    for ind in least_idx:
        if stati[ind].operational:
            least_pending_idx = ind
            break

    num_rows = math.ceil(len(_backends) / 3)

    count = 0
    num_backends = len(_backends)
    for _ in range(num_rows):
        max_len = 0
        str_list = [""] * 8
        for idx in range(3):
            offset = " " * 10 if idx else ""
            config = _backends[count].configuration().to_dict()
            props = _backends[count].properties().to_dict()
            num_qubits = config["n_qubits"]
            str_list[0] += " " * (max_len - len(str_list[0])) + offset
            str_list[0] += _backends[count].name()

            str_list[1] += " " * (max_len - len(str_list[1])) + offset
            str_list[1] += "-" * len(_backends[count].name())

            str_list[2] += " " * (max_len - len(str_list[2])) + offset
            str_list[2] += "Num. Qubits:  %s" % config["n_qubits"]

            str_list[3] += " " * (max_len - len(str_list[3])) + offset
            str_list[3] += "Pending Jobs: %s" % stati[count].pending_jobs

            str_list[4] += " " * (max_len - len(str_list[4])) + offset
            str_list[4] += "Least busy:   %s" % (count == least_pending_idx)

            str_list[5] += " " * (max_len - len(str_list[5])) + offset
            str_list[5] += "Operational:  %s" % stati[count].operational

            str_list[6] += " " * (max_len - len(str_list[6])) + offset
            str_list[6] += "Avg. T1:      %s" % round(
                sum(q[0]["value"] for q in props["qubits"]) / num_qubits, 1
            )
            str_list[7] += " " * (max_len - len(str_list[7])) + offset
            str_list[7] += "Avg. T2:      %s" % round(
                sum(q[1]["value"] for q in props["qubits"]) / num_qubits, 1
            )
            count += 1
            if count == num_backends:
                break
            max_len = max(len(s) for s in str_list)

        print("\n".join(str_list))
        print("\n" * 2)
