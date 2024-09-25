use qiskit_circuit::operations::StandardGate as StdGate;
use crate::xx_decompose::types::{GateData, Circuit2Q};

fn rzx_circuit(theta: f64) -> Circuit2Q {
    let gates = vec! [
        GateData::oneq_no_param(StdGate::HGate, 0),
        GateData::twoq_param(StdGate::RZXGate, theta, 0, 1),
        GateData::oneq_no_param(StdGate::HGate, 0),
    ];
    Circuit2Q::from_gates(gates)
}

fn rzz_circuit(theta: f64) -> Circuit2Q {
    let gates = vec! [
        GateData::oneq_no_param(StdGate::HGate, 0),
        GateData::oneq_no_param(StdGate::HGate, 1),
        GateData::twoq_param(StdGate::RZZGate, theta, 0, 1),
        GateData::oneq_no_param(StdGate::HGate, 0),
        GateData::oneq_no_param(StdGate::HGate, 1),
    ];
    Circuit2Q::from_gates(gates)
}
