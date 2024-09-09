use std::f64::consts::PI;
use std::ops::Index;
use num_complex::Complex64;
use qiskit_circuit::operations::StandardGate;

// One-qubit, one-or-zero parameter, gates
// Only rotation gates and H gate are supported fully.
pub(crate) struct GateData {
    pub gate: StandardGate,
    pub param: Option<f64>,
    pub qubit: i32,
}

impl GateData {
    // TODO: This method works, but is obsolete. Need fewer ways to instantiate.
    pub(crate) fn with_param(gate: StandardGate, param: f64, qubit: i32) -> GateData {
        GateData { gate, param: Some(param), qubit }
    }
}

// A circuit composed of 1Q gates.
// Circuit may have more than one qubit.
pub(crate) struct Circuit1Q {
    gates: Vec<GateData>,
    phase: Complex64,
}

impl Circuit1Q {

    // Reverse the quantum circuit by reversing the order of gates,
    // reflecting the parameter (angle) in rotation gates, and reversing
    // the circuit phase. This is correct for the gates used in this decomposer.
    // This decomposer has only rotation gates and H gates until the last step,
    // at which point we introduce Python and CircuitData.
    fn reverse(&self) -> Circuit1Q {
        let gates: Vec<GateData> = self.gates
            .iter()
            .rev()
            .map(|g| match g {
                // Reverse rotations
                GateData{ gate, param: Some(param), qubit } => GateData { gate: *gate, param: Some(-param), qubit: *qubit },
                // Copy other gates
                GateData{ gate, param: None, qubit } => GateData { gate: *gate, param: None, qubit: *qubit },
            })
            .collect();
        Circuit1Q {gates, phase: -self.phase}
    }
}

// TODO: Need Display for user-facing error message.
#[derive(Debug)]
pub(crate) struct Coordinate {
    data: [f64; 3]
}

impl Coordinate {

    pub(crate) fn reflect(&self, scalars: &[i32; 3]) -> Coordinate {
        Coordinate {
            data: [self.data[0] * (scalars[0]) as f64,
                   self.data[1] * (scalars[1]) as f64,
                   self.data[2] * (scalars[2]) as f64,]
        }
    }

    pub(crate) fn shift(&self, scalars: &[i32; 3]) -> Coordinate {
        let pi2 = PI / 2.0;
        Coordinate {
            data: [pi2 * self.data[0] + (scalars[0]) as f64,
                   pi2 * self.data[1] + (scalars[1]) as f64,
                   pi2 * self.data[2] + (scalars[2]) as f64,]
        }
    }

    /// Unsigned distance between self, axis `i` and other, axis `j`.
    pub(crate) fn distance(&self, other: &Self, i: i32, j: i32) -> f64 {
        let d = self.data[i as usize] - other.data[j as usize];
        (d.abs() % PI).abs()
    }
}

// Forward indexing into `Coordinate` to the field `data`.
impl Index<usize> for Coordinate {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
