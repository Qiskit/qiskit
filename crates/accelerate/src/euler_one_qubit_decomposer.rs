// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

use hashbrown::{HashMap, HashSet};
use num_complex::{Complex64, ComplexFloat};
use smallvec::{smallvec, SmallVec};
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::pybacked::PyBackedStr;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::slice::{PySequenceIndex, SequenceIndex};
use qiskit_circuit::util::c64;
use qiskit_circuit::Qubit;

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;

pub const ANGLE_ZERO_EPSILON: f64 = 1e-12;

#[pyclass(module = "qiskit._accelerate.euler_one_qubit_decomposer")]
pub struct OneQubitGateErrorMap {
    error_map: Vec<HashMap<String, f64>>,
}

#[pymethods]
impl OneQubitGateErrorMap {
    #[new]
    fn new(num_qubits: Option<usize>) -> Self {
        OneQubitGateErrorMap {
            error_map: match num_qubits {
                Some(n) => Vec::with_capacity(n),
                None => Vec::new(),
            },
        }
    }

    fn add_qubit(&mut self, error_map: HashMap<String, f64>) {
        self.error_map.push(error_map);
    }

    fn __getstate__(&self) -> Vec<HashMap<String, f64>> {
        self.error_map.clone()
    }

    fn __setstate__(&mut self, state: Vec<HashMap<String, f64>>) {
        self.error_map = state;
    }
}

#[derive(Debug)]
#[pyclass(sequence)]
pub struct OneQubitGateSequence {
    pub gates: Vec<(StandardGate, SmallVec<[f64; 3]>)>,
    #[pyo3(get)]
    pub global_phase: f64,
}

type OneQubitGateSequenceState = (Vec<(StandardGate, SmallVec<[f64; 3]>)>, f64);

#[pymethods]
impl OneQubitGateSequence {
    #[new]
    fn new() -> Self {
        OneQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }
    fn __getstate__(&self) -> OneQubitGateSequenceState {
        (self.gates.clone(), self.global_phase)
    }

    fn __setstate__(&mut self, state: OneQubitGateSequenceState) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gates.len())
    }

    fn __getitem__(&self, py: Python, idx: PySequenceIndex) -> PyResult<PyObject> {
        match idx.with_len(self.gates.len())? {
            SequenceIndex::Int(idx) => Ok(self.gates[idx].to_object(py)),
            indices => Ok(PyList::new_bound(
                py,
                indices.iter().map(|pos| self.gates[pos].to_object(py)),
            )
            .into_any()
            .unbind()),
        }
    }
}

fn circuit_kak(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    k_gate: StandardGate,
    a_gate: StandardGate,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut lam = lam;
    let mut theta = theta;
    let mut phi = phi;
    let mut circuit: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::with_capacity(3);
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    let mut global_phase = phase - (phi + lam) / 2.;
    if theta.abs() < atol {
        lam += phi;
        // NOTE: The following normalization is safe, because the gphase correction below
        // fixes a particular diagonal entry to 1, which prevents any potential phase
        // slippage coming from _mod_2pi injecting multiples of 2pi.
        lam = mod_2pi(lam, atol);
        if lam.abs() > atol {
            circuit.push((k_gate, smallvec![lam]));
            global_phase += lam / 2.;
        }
        return OneQubitGateSequence {
            gates: circuit,
            global_phase,
        };
    }
    if (theta - PI).abs() < atol {
        global_phase += phi;
        lam -= phi;
        phi = 0.;
    }
    if mod_2pi(lam + PI, atol).abs() < atol || mod_2pi(phi + PI, atol).abs() < atol {
        lam += PI;
        theta = -theta;
        phi += PI;
    }
    lam = mod_2pi(lam, atol);
    if lam.abs() > atol {
        global_phase += lam / 2.;
        circuit.push((k_gate, smallvec![lam]));
    }
    circuit.push((a_gate, smallvec![theta]));
    phi = mod_2pi(phi, atol);
    if phi.abs() > atol {
        global_phase += phi / 2.;
        circuit.push((k_gate, smallvec![phi]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase,
    }
}

fn circuit_u3(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    let phi = mod_2pi(phi, atol);
    let lam = mod_2pi(lam, atol);
    if !simplify || theta.abs() > atol || phi.abs() > atol || lam.abs() > atol {
        circuit.push((StandardGate::U3Gate, smallvec![theta, phi, lam]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_u321(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    if theta.abs() < atol {
        let tot = mod_2pi(phi + lam, atol);
        if tot.abs() > atol {
            circuit.push((StandardGate::U1Gate, smallvec![tot]));
        }
    } else if (theta - PI / 2.).abs() < atol {
        circuit.push((
            StandardGate::U2Gate,
            smallvec![mod_2pi(phi, atol), mod_2pi(lam, atol)],
        ));
    } else {
        circuit.push((
            StandardGate::U3Gate,
            smallvec![theta, mod_2pi(phi, atol), mod_2pi(lam, atol)],
        ));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_u(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    let phi = mod_2pi(phi, atol);
    let lam = mod_2pi(lam, atol);
    if theta.abs() > atol || phi.abs() > atol || lam.abs() > atol {
        circuit.push((StandardGate::UGate, smallvec![theta, phi, lam]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_psx_gen<F, P, X>(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
    mut pfun: P,
    mut xfun: F,
    xpifun: Option<X>,
) -> OneQubitGateSequence
where
    F: FnMut(&mut OneQubitGateSequence),
    P: FnMut(&mut OneQubitGateSequence, f64),
    X: FnOnce(&mut OneQubitGateSequence),
{
    let mut phi = phi;
    let mut lam = lam;
    let mut theta = theta;
    let mut circuit = OneQubitGateSequence {
        gates: Vec::new(),
        global_phase: phase,
    };
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    // Early return for zero SX decomposition
    if theta.abs() < atol {
        pfun(&mut circuit, lam + phi);
        return circuit;
    }
    // Early return for single SX decomposition
    if (theta - PI / 2.).abs() < atol {
        pfun(&mut circuit, lam - PI / 2.);
        xfun(&mut circuit);
        pfun(&mut circuit, phi + PI / 2.);
        return circuit;
    }
    // General double SX decomposition
    if (theta - PI).abs() < atol {
        circuit.global_phase += lam;
        phi -= lam;
        lam = 0.;
    }
    if mod_2pi(lam + PI, atol).abs() < atol || mod_2pi(phi, atol).abs() < atol {
        lam += PI;
        theta = -theta;
        phi += PI;
        circuit.global_phase -= theta;
    }
    // Shift theta and phi to turn the decomposition from
    // RZ(phi).RY(theta).RZ(lam) = RZ(phi).RX(-pi/2).RZ(theta).RX(pi/2).RZ(lam)
    // into RZ(phi+pi).SX.RZ(theta+pi).SX.RZ(lam).
    theta += PI;
    phi += PI;
    circuit.global_phase -= PI / 2.;
    // emit circuit
    pfun(&mut circuit, lam);
    match xpifun {
        Some(xpifun) if mod_2pi(theta, atol).abs() < atol => xpifun(&mut circuit),
        _ => {
            xfun(&mut circuit);
            pfun(&mut circuit, theta);
            xfun(&mut circuit);
        }
    };
    pfun(&mut circuit, phi);
    circuit
}

fn circuit_rr(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }

    if mod_2pi((phi + lam) / 2., atol).abs() < atol {
        // This can be expressed as a single R gate
        if theta.abs() > atol {
            circuit.push((
                StandardGate::RGate,
                smallvec![theta, mod_2pi(PI / 2. + phi, atol)],
            ));
        }
    } else {
        // General case: use two R gates
        if (theta - PI).abs() > atol {
            circuit.push((
                StandardGate::RGate,
                smallvec![theta - PI, mod_2pi(PI / 2. - lam, atol)],
            ));
        }
        circuit.push((
            StandardGate::RGate,
            smallvec![PI, mod_2pi(0.5 * (phi - lam + PI), atol)],
        ));
    }

    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

#[pyfunction]
pub fn generate_circuit(
    target_basis: &EulerBasis,
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<OneQubitGateSequence> {
    let res = match target_basis {
        EulerBasis::ZYZ => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RZGate,
            StandardGate::RYGate,
            simplify,
            atol,
        ),
        EulerBasis::ZXZ => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RZGate,
            StandardGate::RXGate,
            simplify,
            atol,
        ),
        EulerBasis::XZX => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RXGate,
            StandardGate::RZGate,
            simplify,
            atol,
        ),
        EulerBasis::XYX => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RXGate,
            StandardGate::RYGate,
            simplify,
            atol,
        ),
        EulerBasis::U3 => circuit_u3(theta, phi, lam, phase, simplify, atol),
        EulerBasis::U321 => circuit_u321(theta, phi, lam, phase, simplify, atol),
        EulerBasis::U => circuit_u(theta, phi, lam, phase, simplify, atol),
        EulerBasis::PSX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit
                        .gates
                        .push((StandardGate::PhaseGate, smallvec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SXGate, SmallVec::new()));
            };

            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::ZSX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::RZGate, smallvec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SXGate, SmallVec::new()));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::U1X => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::U1Gate, smallvec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.global_phase += PI / 4.;
                circuit
                    .gates
                    .push((StandardGate::RXGate, smallvec![PI / 2.]));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::ZSXX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::RZGate, smallvec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SXGate, SmallVec::new()));
            };
            let fnxpi = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::XGate, SmallVec::new()));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                Some(fnxpi),
            )
        }
        EulerBasis::RR => circuit_rr(theta, phi, lam, phase, simplify, atol),
    };
    Ok(res)
}

const EULER_BASIS_SIZE: usize = 12;

static EULER_BASES: [&[&str]; EULER_BASIS_SIZE] = [
    &["u3"],
    &["u3", "u2", "u1"],
    &["u"],
    &["p", "sx"],
    &["u1", "rx"],
    &["r"],
    &["rz", "ry"],
    &["rz", "rx"],
    &["rz", "rx"],
    &["rx", "ry"],
    &["rz", "sx", "x"],
    &["rz", "sx"],
];
static EULER_BASIS_NAMES: [EulerBasis; EULER_BASIS_SIZE] = [
    EulerBasis::U3,
    EulerBasis::U321,
    EulerBasis::U,
    EulerBasis::PSX,
    EulerBasis::U1X,
    EulerBasis::RR,
    EulerBasis::ZYZ,
    EulerBasis::ZXZ,
    EulerBasis::XZX,
    EulerBasis::XYX,
    EulerBasis::ZSXX,
    EulerBasis::ZSX,
];

/// A structure containing a set of supported `EulerBasis` for running 1q synthesis
#[derive(Debug, Clone)]
pub struct EulerBasisSet {
    basis: [bool; EULER_BASIS_SIZE],
    initialized: bool,
}

impl EulerBasisSet {
    // Instantiate a new EulerBasisSet
    pub fn new() -> Self {
        EulerBasisSet {
            basis: [false; EULER_BASIS_SIZE],
            initialized: false,
        }
    }

    /// Return true if this has been initialized any basis is supported
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Add a basis to the set
    pub fn add_basis(&mut self, basis: EulerBasis) {
        self.basis[basis as usize] = true;
        self.initialized = true;
    }

    /// Get an iterator of all the supported EulerBasis
    pub fn get_bases(&self) -> impl Iterator<Item = EulerBasis> + '_ {
        self.basis
            .iter()
            .enumerate()
            .filter_map(|(index, supported)| {
                if *supported {
                    Some(EULER_BASIS_NAMES[index])
                } else {
                    None
                }
            })
    }

    /// Check if a basis is supported by this set
    pub fn basis_supported(&self, basis: EulerBasis) -> bool {
        self.basis[basis as usize]
    }

    /// Modify this set to support all EulerBasis
    pub fn support_all(&mut self) {
        self.basis = [true; 12];
        self.initialized = true;
    }

    /// Remove an basis from the set
    pub fn remove(&mut self, basis: EulerBasis) {
        self.basis[basis as usize] = false;
    }
}

impl Default for EulerBasisSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Copy, Eq, Hash, PartialEq)]
#[pyclass(module = "qiskit._accelerate.euler_one_qubit_decomposer")]
pub enum EulerBasis {
    U3 = 0,
    U321 = 1,
    U = 2,
    PSX = 3,
    U1X = 4,
    RR = 5,
    ZYZ = 6,
    ZXZ = 7,
    XZX = 8,
    XYX = 9,
    ZSXX = 10,
    ZSX = 11,
}

impl EulerBasis {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U321 => "U321",
            Self::U3 => "U3",
            Self::U => "U",
            Self::PSX => "PSX",
            Self::ZSX => "ZSX",
            Self::ZSXX => "ZSXX",
            Self::U1X => "U1X",
            Self::RR => "RR",
            Self::ZYZ => "ZYZ",
            Self::ZXZ => "ZXZ",
            Self::XYX => "XYX",
            Self::XZX => "XZX",
        }
    }
}

#[pymethods]
impl EulerBasis {
    fn __reduce__(&self, py: Python) -> Py<PyAny> {
        (
            py.get_type_bound::<Self>(),
            (PyString::new_bound(py, self.as_str()),),
        )
            .into_py(py)
    }

    #[new]
    pub fn __new__(input: &str) -> PyResult<Self> {
        Self::from_str(input)
            .map_err(|_| PyValueError::new_err(format!("Invalid target basis '{input}'")))
    }
}

impl FromStr for EulerBasis {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "U321" => Ok(EulerBasis::U321),
            "U3" => Ok(EulerBasis::U3),
            "U" => Ok(EulerBasis::U),
            "PSX" => Ok(EulerBasis::PSX),
            "ZSX" => Ok(EulerBasis::ZSX),
            "ZSXX" => Ok(EulerBasis::ZSXX),
            "U1X" => Ok(EulerBasis::U1X),
            "RR" => Ok(EulerBasis::RR),
            "ZYZ" => Ok(EulerBasis::ZYZ),
            "ZXZ" => Ok(EulerBasis::ZXZ),
            "XYX" => Ok(EulerBasis::XYX),
            "XZX" => Ok(EulerBasis::XZX),
            _ => Err(()),
        }
    }
}

#[inline]
pub fn angles_from_unitary(unitary: ArrayView2<Complex64>, target_basis: EulerBasis) -> [f64; 4] {
    match target_basis {
        EulerBasis::U321 => params_u3_inner(unitary),
        EulerBasis::U3 => params_u3_inner(unitary),
        EulerBasis::U => params_u3_inner(unitary),
        EulerBasis::PSX => params_u1x_inner(unitary),
        EulerBasis::ZSX => params_u1x_inner(unitary),
        EulerBasis::ZSXX => params_u1x_inner(unitary),
        EulerBasis::U1X => params_u1x_inner(unitary),
        EulerBasis::RR => params_zyz_inner(unitary),
        EulerBasis::ZYZ => params_zyz_inner(unitary),
        EulerBasis::ZXZ => params_zxz_inner(unitary),
        EulerBasis::XYX => params_xyx_inner(unitary),
        EulerBasis::XZX => params_xzx_inner(unitary),
    }
}

#[inline]
fn compare_error_fn(
    circuit: &OneQubitGateSequence,
    error_map: &Option<&OneQubitGateErrorMap>,
    qubit: usize,
) -> (f64, usize) {
    match error_map {
        Some(global_err_map) => {
            let err_map = &global_err_map.error_map[qubit];
            let fidelity_product: f64 = circuit
                .gates
                .iter()
                .map(|gate| 1. - err_map.get(gate.0.name()).unwrap_or(&0.))
                .product();
            (1. - fidelity_product, circuit.gates.len())
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

fn compute_error_term(gate: &str, error_map: &OneQubitGateErrorMap, qubit: usize) -> f64 {
    1. - error_map.error_map[qubit].get(gate).unwrap_or(&0.)
}

fn compute_error_str(
    gates: &[(String, SmallVec<[f64; 3]>)],
    error_map: Option<&OneQubitGateErrorMap>,
    qubit: usize,
) -> (f64, usize) {
    match error_map {
        Some(err_map) => {
            let num_gates = gates.len();
            let gate_fidelities: f64 = gates
                .iter()
                .map(|gate| compute_error_term(gate.0.as_str(), err_map, qubit))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (gates.len() as f64, gates.len()),
    }
}

#[pyfunction]
pub fn compute_error_list(
    circuit: Vec<PyRef<DAGOpNode>>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
) -> (f64, usize) {
    let circuit_list: Vec<(String, SmallVec<[f64; 3]>)> = circuit
        .iter()
        .map(|node| {
            (
                node.instruction.operation.name().to_string(),
                smallvec![], // Params not needed in this path
            )
        })
        .collect();
    compute_error_str(&circuit_list, error_map, qubit)
}

#[pyfunction]
#[pyo3(signature = (unitary, target_basis_list, qubit, error_map=None, simplify=true, atol=None))]
pub fn unitary_to_gate_sequence(
    unitary: PyReadonlyArray2<Complex64>,
    target_basis_list: Vec<PyBackedStr>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<Option<OneQubitGateSequence>> {
    let mut target_basis_set = EulerBasisSet::new();
    for basis in target_basis_list
        .iter()
        .map(|basis| EulerBasis::__new__(basis))
    {
        target_basis_set.add_basis(basis?);
    }
    Ok(unitary_to_gate_sequence_inner(
        unitary.as_array(),
        &target_basis_set,
        qubit,
        error_map,
        simplify,
        atol,
    ))
}

#[inline]
pub fn unitary_to_gate_sequence_inner(
    unitary_mat: ArrayView2<Complex64>,
    target_basis_list: &EulerBasisSet,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> Option<OneQubitGateSequence> {
    target_basis_list
        .get_bases()
        .map(|target_basis| {
            let [theta, phi, lam, phase] = angles_from_unitary(unitary_mat, target_basis);
            generate_circuit(&target_basis, theta, phi, lam, phase, simplify, atol).unwrap()
        })
        .min_by(|a, b| {
            let error_a = compare_error_fn(a, &error_map, qubit);
            let error_b = compare_error_fn(b, &error_map, qubit);
            error_a.partial_cmp(&error_b).unwrap_or(Ordering::Equal)
        })
}

#[pyfunction]
#[pyo3(signature = (unitary, target_basis_list, qubit, error_map=None, simplify=true, atol=None))]
pub fn unitary_to_circuit(
    py: Python,
    unitary: PyReadonlyArray2<Complex64>,
    target_basis_list: Vec<PyBackedStr>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<Option<CircuitData>> {
    let mut target_basis_set = EulerBasisSet::new();
    for basis in target_basis_list
        .iter()
        .map(|basis| EulerBasis::__new__(basis))
    {
        target_basis_set.add_basis(basis?);
    }
    let circuit_sequence = unitary_to_gate_sequence_inner(
        unitary.as_array(),
        &target_basis_set,
        qubit,
        error_map,
        simplify,
        atol,
    );
    Ok(circuit_sequence.map(|seq| {
        CircuitData::from_standard_gates(
            py,
            1,
            seq.gates.into_iter().map(|(gate, params)| {
                (
                    gate,
                    params.into_iter().map(Param::Float).collect(),
                    smallvec![Qubit(0)],
                )
            }),
            Param::Float(seq.global_phase),
        )
        .expect("Unexpected Qiskit python bug")
    }))
}

#[inline]
pub fn det_one_qubit(mat: ArrayView2<Complex64>) -> Complex64 {
    mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]]
}

/// Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π
#[inline]
fn mod_2pi(angle: f64, atol: f64) -> f64 {
    // f64::rem_euclid() isn't exactly the same as Python's % operator, but because
    // the RHS here is a constant and positive it is effectively equivalent for
    // this case
    let wrapped = (angle + PI).rem_euclid(2. * PI) - PI;
    if (wrapped - PI).abs() < atol {
        -PI
    } else {
        wrapped
    }
}

fn params_zyz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let det_arg = det_one_qubit(mat).arg();
    let phase = 0.5 * det_arg;
    let theta = 2. * mat[[1, 0]].abs().atan2(mat[[0, 0]].abs());
    let ang1 = mat[[1, 1]].arg();
    let ang2 = mat[[1, 0]].arg();
    let phi = ang1 + ang2 - det_arg;
    let lam = ang1 - ang2;
    [theta, phi, lam, phase]
}

fn params_zxz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi + PI / 2., lam - PI / 2., phase]
}

#[pyfunction]
pub fn params_zyz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zyz_inner(mat)
}

fn params_u3_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    // The determinant of U3 gate depends on its params
    // via det(u3(theta, phi, lam)) = exp(1j*(phi+lam))
    // Since the phase is wrt to a SU matrix we must rescale
    // phase to correct this
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi, lam, phase - 0.5 * (phi + lam)]
}

#[pyfunction]
pub fn params_u3(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_u3_inner(mat)
}

fn params_u1x_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    // The determinant of this decomposition depends on its params
    // Since the phase is wrt to a SU matrix we must rescale
    // phase to correct this
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi, lam, phase - 0.5 * (theta + phi + lam)]
}

#[pyfunction]
pub fn params_u1x(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_u1x_inner(mat)
}

fn params_xyx_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let mat_zyz = arr2(&[
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] + mat[[1, 0]] + mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] + mat[[1, 0]] - mat[[1, 1]]),
        ],
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] - mat[[1, 0]] - mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] - mat[[1, 0]] + mat[[1, 1]]),
        ],
    ]);
    let [theta, phi, lam, phase] = params_zyz_inner(mat_zyz.view());
    let new_phi = mod_2pi(phi + PI, 0.);
    let new_lam = mod_2pi(lam + PI, 0.);
    [
        theta,
        new_phi,
        new_lam,
        phase + (new_phi + new_lam - phi - lam) / 2.,
    ]
}

#[pyfunction]
pub fn params_xyx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_xyx_inner(mat)
}

fn params_xzx_inner(umat: ArrayView2<Complex64>) -> [f64; 4] {
    let det = det_one_qubit(umat);
    let phase = det.ln().im / 2.;
    let sqrt_det = det.sqrt();
    let mat_zyz = arr2(&[
        [
            c64((umat[[0, 0]] / sqrt_det).re, (umat[[1, 0]] / sqrt_det).im),
            c64((umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
        ],
        [
            c64(-(umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
            c64((umat[[0, 0]] / sqrt_det).re, -(umat[[1, 0]] / sqrt_det).im),
        ],
    ]);
    let [theta, phi, lam, phase_zxz] = params_zxz_inner(mat_zyz.view());
    [theta, phi, lam, phase + phase_zxz]
}

#[pyfunction]
pub fn params_xzx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let umat = unitary.as_array();
    params_xzx_inner(umat)
}

#[pyfunction]
pub fn params_zxz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zxz_inner(mat)
}

fn compute_error_term_from_target(gate: &str, target: &Target, qubit: PhysicalQubit) -> f64 {
    1. - target.get_error(gate, &[qubit]).unwrap_or(0.)
}

fn compute_error_from_target_one_qubit_sequence(
    circuit: &OneQubitGateSequence,
    qubit: PhysicalQubit,
    target: Option<&Target>,
) -> (f64, usize) {
    match target {
        Some(target) => {
            let num_gates = circuit.gates.len();
            let gate_fidelities: f64 = circuit
                .gates
                .iter()
                .map(|gate| compute_error_term_from_target(gate.0.name(), target, qubit))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

#[pyfunction]
#[pyo3(signature = (dag, *, target=None, basis_gates=None, global_decomposers=None))]
pub(crate) fn optimize_1q_gates_decomposition(
    py: Python,
    dag: &mut DAGCircuit,
    target: Option<&Target>,
    basis_gates: Option<HashSet<String>>,
    global_decomposers: Option<Vec<String>>,
) -> PyResult<()> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();
    let dag_qubits = dag.num_qubits();
    let mut target_basis_per_qubit: Vec<EulerBasisSet> = vec![EulerBasisSet::new(); dag_qubits];
    let mut basis_gates_per_qubit: Vec<Option<HashSet<&str>>> = vec![None; dag_qubits];
    for raw_run in runs {
        let mut error = match target {
            Some(_) => 1.,
            None => raw_run.len() as f64,
        };
        let qubit: PhysicalQubit = if let NodeType::Operation(inst) = &dag.dag[raw_run[0]] {
            PhysicalQubit::new(dag.get_qargs(inst.qubits)[0].0)
        } else {
            unreachable!("nodes in runs will always be op nodes")
        };
        if !dag.calibrations_empty() {
            let mut has_calibration = false;
            for node in &raw_run {
                if dag.has_calibration_for_index(py, *node)? {
                    has_calibration = true;
                    break;
                }
            }
            if has_calibration {
                continue;
            }
        }
        if basis_gates_per_qubit[qubit.0 as usize].is_none() {
            let basis_gates = match target {
                Some(target) => Some(
                    target
                        .operation_names_for_qargs(Some(&smallvec![qubit]))
                        .unwrap(),
                ),
                None => {
                    let basis = basis_gates.as_ref();
                    basis.map(|basis| basis.iter().map(|x| x.as_str()).collect())
                }
            };
            basis_gates_per_qubit[qubit.0 as usize] = basis_gates;
        }
        let basis_gates = &basis_gates_per_qubit[qubit.0 as usize].as_ref();

        let target_basis_set = &mut target_basis_per_qubit[qubit.0 as usize];
        if !target_basis_set.initialized() {
            match target {
                Some(_target) => EULER_BASES
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, gates)| {
                        if !gates
                            .iter()
                            .all(|gate| basis_gates.as_ref().unwrap().contains(gate))
                        {
                            return None;
                        }
                        let basis = EULER_BASIS_NAMES[idx];
                        Some(basis)
                    })
                    .for_each(|basis| target_basis_set.add_basis(basis)),
                None => match &global_decomposers {
                    Some(bases) => bases
                        .iter()
                        .map(|basis| EulerBasis::__new__(basis).unwrap())
                        .for_each(|basis| target_basis_set.add_basis(basis)),
                    None => match basis_gates {
                        Some(gates) => EULER_BASES
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, basis_gates)| {
                                if !gates.iter().all(|gate| basis_gates.as_ref().contains(gate)) {
                                    return None;
                                }
                                let basis = EULER_BASIS_NAMES[idx];
                                Some(basis)
                            })
                            .for_each(|basis| target_basis_set.add_basis(basis)),
                        None => target_basis_set.support_all(),
                    },
                },
            };
            if target_basis_set.basis_supported(EulerBasis::U3)
                && target_basis_set.basis_supported(EulerBasis::U321)
            {
                target_basis_set.remove(EulerBasis::U3);
            }
            if target_basis_set.basis_supported(EulerBasis::ZSX)
                && target_basis_set.basis_supported(EulerBasis::ZSXX)
            {
                target_basis_set.remove(EulerBasis::ZSX);
            }
        }
        let target_basis_set = &target_basis_per_qubit[qubit.0 as usize];
        let operator = raw_run
            .iter()
            .map(|node_index| {
                let node = &dag.dag[*node_index];
                if let NodeType::Operation(inst) = node {
                    if let Some(target) = target {
                        error *= compute_error_term_from_target(inst.op.name(), target, qubit);
                    }
                    inst.op.matrix(inst.params_view()).unwrap()
                } else {
                    unreachable!("Can only have op nodes here")
                }
            })
            .fold(
                [
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                ],
                |mut operator, node| {
                    matmul_1q(&mut operator, node);
                    operator
                },
            );

        let old_error = if target.is_some() {
            (1. - error, raw_run.len())
        } else {
            (error, raw_run.len())
        };
        let sequence = unitary_to_gate_sequence_inner(
            aview2(&operator),
            target_basis_set,
            qubit.0 as usize,
            None,
            true,
            None,
        );
        let sequence = match sequence {
            Some(seq) => seq,
            None => continue,
        };
        let new_error = compute_error_from_target_one_qubit_sequence(&sequence, qubit, target);

        let mut outside_basis = false;
        if let Some(basis) = basis_gates {
            for node in &raw_run {
                if let NodeType::Operation(inst) = &dag.dag[*node] {
                    if !basis.contains(inst.op.name()) {
                        outside_basis = true;
                        break;
                    }
                }
            }
        } else {
            outside_basis = false;
        }
        if outside_basis
            || new_error < old_error
            || new_error.0.abs() < 1e-9 && old_error.0.abs() >= 1e-9
        {
            for gate in sequence.gates {
                dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
            }
            dag.add_global_phase(py, &Param::Float(sequence.global_phase))?;
            dag.remove_1q_sequence(&raw_run);
        }
    }
    Ok(())
}

fn matmul_1q(operator: &mut [[Complex64; 2]; 2], other: Array2<Complex64>) {
    *operator = [
        [
            other[[0, 0]] * operator[0][0] + other[[0, 1]] * operator[1][0],
            other[[0, 0]] * operator[0][1] + other[[0, 1]] * operator[1][1],
        ],
        [
            other[[1, 0]] * operator[0][0] + other[[1, 1]] * operator[1][0],
            other[[1, 0]] * operator[0][1] + other[[1, 1]] * operator[1][1],
        ],
    ];
}

pub fn euler_one_qubit_decomposer(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(params_zyz))?;
    m.add_wrapped(wrap_pyfunction!(params_xyx))?;
    m.add_wrapped(wrap_pyfunction!(params_xzx))?;
    m.add_wrapped(wrap_pyfunction!(params_zxz))?;
    m.add_wrapped(wrap_pyfunction!(params_u3))?;
    m.add_wrapped(wrap_pyfunction!(params_u1x))?;
    m.add_wrapped(wrap_pyfunction!(generate_circuit))?;
    m.add_wrapped(wrap_pyfunction!(unitary_to_gate_sequence))?;
    m.add_wrapped(wrap_pyfunction!(unitary_to_circuit))?;
    m.add_wrapped(wrap_pyfunction!(compute_error_list))?;
    m.add_wrapped(wrap_pyfunction!(optimize_1q_gates_decomposition))?;
    m.add_class::<OneQubitGateSequence>()?;
    m.add_class::<OneQubitGateErrorMap>()?;
    m.add_class::<EulerBasis>()?;
    Ok(())
}
