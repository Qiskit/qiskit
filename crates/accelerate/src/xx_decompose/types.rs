use std::f64::consts::PI;
use std::ops::Index;
use num_complex::Complex64;
use qiskit_circuit::operations::StandardGate as StdGate;
use crate::xx_decompose::utilities::EPSILON;

// Represent gates with:
// * One or two qubits
// * Zero or one (angle) parameters
// `GateData` is meant to support two-qubit circuits. So qubit indices take values
//  zero and one. There are a lot of ways to encode the information:
//  * Distinguish one or two qubit gate
//  * Indexes of first and second qubits.
// We could do this in three bits if need be. We´ll try to hide the implementation.
#[derive(Copy, Clone)]
pub(crate) struct GateData {
    pub gate: StdGate,
    pub param: Option<f64>,
    pub qubit0: i32, // Takes values zero and one
    pub qubit1: Option<i32>,
}

impl GateData {

    pub(crate) fn oneq_no_param(gate: StdGate, qubit: i32) -> GateData {
        GateData {gate, param: None, qubit0: qubit, qubit1: None }
    }

    pub(crate) fn oneq_param(gate: StdGate, param: f64, qubit: i32) -> GateData {
        GateData {gate, param: Some(param), qubit0: qubit, qubit1: None }
    }

    pub(crate) fn twoq_no_param(gate: StdGate, qubit0: i32, qubit1: i32) -> GateData {
        GateData {gate, param: None, qubit0, qubit1: Some(qubit1) }
    }

    pub(crate) fn twoq_param(gate: StdGate, param: f64, qubit0: i32, qubit1: i32) -> GateData {
        GateData {gate, param: Some(param), qubit0, qubit1: Some(qubit1) }
    }

    // TODO: what kind of name is good for predicates?
    pub(crate) fn has_param(&self) -> bool {
        self.param.is_some()
    }

    pub(crate) fn get_param(&self) -> f64 {
        self.param.unwrap()
    }

    pub(crate) fn get_name(&self) -> StdGate {
        self.gate
    }

    pub(crate) fn is_oneq(&self) -> bool {
        self.qubit1.is_none()
    }

    pub(crate) fn is_twoq(&self) -> bool {
        ! self.is_oneq()
    }

    pub(crate) fn reverse(&self) -> GateData {
        let mut gate = self.clone();
        if self.has_param() {
            gate.param = Some(-self.get_param());
            return gate
        }
        let gate_name = match self.get_name() {
            StdGate::HGate |
            StdGate::XGate | StdGate::YGate | StdGate::ZGate |
            StdGate::CXGate | StdGate::CYGate | StdGate::CZGate |
            StdGate::CHGate
                => self.get_name(),
            StdGate::SGate => StdGate::SdgGate,
            StdGate::TGate => StdGate::TdgGate,
            StdGate::SXGate => StdGate::SXdgGate,
            _ => panic!("No support for this gate"),
        };
        gate.gate = gate_name;
        gate
    }
}

// A two-qubit circuit composed of one- and two-qubit gates
pub(crate) struct Circuit2Q {
    gates: Vec<GateData>,
    phase: Complex64,
}

impl Circuit2Q {

    // Reverse the quantum circuit by reversing the order of gates,
    // reflecting the parameter (angle) in rotation gates, and reversing
    // the circuit phase. This is correct for the gates used in this decomposer.
    // This decomposer has only rotation gates and H gates until the last step,
    // at which point we introduce Python and CircuitData.
    fn reverse(&self) -> Circuit2Q {
        let gates: Vec<GateData> = self.gates
            .iter()
            .rev()
            .map(|g| g.reverse())
            .collect();
        Circuit2Q {gates, phase: -self.phase}
    }

    pub(crate) fn from_gates(gates: Vec<GateData>) -> Circuit2Q {
        Circuit2Q { gates, phase: Complex64::new(0.0, 0.0) }
    }
}

// TODO: Need Display for user-facing error message.
#[derive(Debug, Copy, Clone)]
pub(crate) struct Coordinate {
    data: [f64; 3]
}

impl Coordinate {

    pub(crate) fn reflect0(&mut self) {
        self.data[0] = PI / 2. -  self.data[0];
    }

    pub(crate) fn needs_reflect0(&self) -> bool {
        if self.data[0] >= PI / 4. + EPSILON {
            return true
        }
        false
    }

    pub(crate) fn reflect(&self, scalars: &[f64; 3]) -> Coordinate {
        Coordinate {
            data: [self.data[0] * (scalars[0]),
                   self.data[1] * (scalars[1]),
                   self.data[2] * (scalars[2]),]
        }
    }

    pub(crate) fn shift(&self, scalars: &[f64; 3]) -> Coordinate {
        let pi2 = PI / 2.0;
        Coordinate {
            data: [pi2 * self.data[0] + (scalars[0]),
                   pi2 * self.data[1] + (scalars[1]),
                   pi2 * self.data[2] + (scalars[2]),]
        }
    }

    /// Unsigned distance between self, axis `i` and other, axis `j`.
    pub(crate) fn distance(&self, other: &Self, i: i32, j: i32) -> f64 {
        let d = self.data[i as usize] - other.data[j as usize];
        (d.abs() % PI).abs()
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }
}

// Forward indexing into `Coordinate` to the field `data`.
impl Index<usize> for Coordinate {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
