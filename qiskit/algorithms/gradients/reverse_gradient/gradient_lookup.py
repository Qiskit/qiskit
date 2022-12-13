# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Split a circuit into subcircuits, each containing a single parameterized gate."""

import itertools
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate


def extract_single_parameter(expression):
    if isinstance(expression, Parameter):
        return expression

    if len(expression.parameters) > 1:
        raise ValueError("Expression has more than one parameter.")

    return list(expression.parameters)[0]


def gradient_lookup(gate):
    """Returns a circuit implementing the gradient of the input gate."""

    param = gate.params[0]
    if isinstance(gate, RXGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rx(param, 0)
        derivative.x(0)
        return [[-0.5j, derivative]]
    if isinstance(gate, RYGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.ry(param, 0)
        derivative.y(0)
        return [[-0.5j, derivative]]
    if isinstance(gate, RZGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rz(param, 0)
        derivative.z(0)
        return [[-0.5j, derivative]]
    if isinstance(gate, CRXGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rx(param, 1)
        proj1.x(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rx(param, 1)
        proj2.x(1)

        return [[-0.25j, proj1], [0.25j, proj2]]
    if isinstance(gate, CRYGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.ry(param, 1)
        proj1.y(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.ry(param, 1)
        proj2.y(1)

        return [[-0.25j, proj1], [0.25j, proj2]]
    if isinstance(gate, CRZGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rz(param, 1)
        proj1.z(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rz(param, 1)
        proj2.z(1)

        return [[-0.25j, proj1], [0.25j, proj2]]
    raise NotImplementedError("Cannot implement for", gate)


def analytic_gradient(circuit, parameter=None, return_parameter=False):
    """Return the analytic gradient of the input circuit."""

    if parameter is not None:
        single = extract_single_parameter(parameter)

        if single not in circuit.parameters:
            raise ValueError("Parameter not in this circuit.")

        if len(circuit._parameter_table[single]) > 1:
            raise NotImplementedError("No product rule support yet, params must be unique.")

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate = op[0]
        op_context += [op[1:]]
        if (parameter is None and len(gate.params) > 0) or parameter in gate.params:
            coeffs_and_grads = gradient_lookup(gate)
            if not isinstance(parameter, Parameter):
                # is not a fully decomposed parameter
                if len(parameter.parameters) > 1:
                    raise NotImplementedError("Cannot support multiple parameters in one gate yet.")
                single_parameter = list(parameter.parameters)[0]

                # multiply coefficient with parameter derivative
                for k, coeff_and_grad in enumerate(coeffs_and_grads):
                    coeffs_and_grads[k][0] = (
                        parameter.gradient(single_parameter) * coeff_and_grad[0]
                    )

            summands += [coeffs_and_grads]
        else:
            summands += [[[1, gate]]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        c = 1
        for i, a in enumerate(product_rule_term):
            c *= a[0]
            summand_circuit.data.append([a[1], *op_context[i]])
        gradient += [[c, summand_circuit.copy()]]

    return gradient
