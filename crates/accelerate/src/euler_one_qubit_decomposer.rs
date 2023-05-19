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

use hashbrown::HashMap;
use num_complex::{Complex64, ComplexFloat};
use std::cmp::Ordering;
use std::f64::consts::PI;

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use ndarray::prelude::*;
use numpy::PyReadonlyArray2;

const DEFAULT_ATOL: f64 = 1e-12;

#[derive(FromPyObject)]
enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

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

#[pyclass(sequence)]
pub struct OneQubitGateSequence {
    gates: Vec<(String, Vec<f64>)>,
    #[pyo3(get)]
    global_phase: f64,
}

#[pymethods]
impl OneQubitGateSequence {
    #[new]
    fn new() -> Self {
        OneQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }
    fn __getstate__(&self) -> (Vec<(String, Vec<f64>)>, f64) {
        (self.gates.clone(), self.global_phase)
    }

    fn __setstate__(&mut self, state: (Vec<(String, Vec<f64>)>, f64)) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gates.len())
    }

    fn __getitem__(&self, py: Python, idx: SliceOrInt) -> PyResult<PyObject> {
        match idx {
            SliceOrInt::Slice(slc) => {
                let len = self.gates.len().try_into().unwrap();
                let indices = slc.indices(len)?;
                let mut out_vec: Vec<(String, Vec<f64>)> = Vec::new();
                // Start and stop will always be positive the slice api converts
                // negatives to the index for example:
                // list(range(5))[-1:-3:-1]
                // will return start=4, stop=2, and step=-1
                let mut pos: isize = indices.start;
                let mut cond = if indices.step < 0 {
                    pos > indices.stop
                } else {
                    pos < indices.stop
                };
                while cond {
                    if pos < len as isize {
                        out_vec.push(self.gates[pos as usize].clone());
                    }
                    pos += indices.step;
                    if indices.step < 0 {
                        cond = pos > indices.stop;
                    } else {
                        cond = pos < indices.stop;
                    }
                }
                Ok(out_vec.into_py(py))
            }
            SliceOrInt::Int(idx) => {
                let len = self.gates.len() as isize;
                if idx >= len || idx < -len {
                    Err(PyIndexError::new_err(format!("Invalid index, {idx}")))
                } else if idx < 0 {
                    let len = self.gates.len();
                    Ok(self.gates[len - idx.unsigned_abs()].to_object(py))
                } else {
                    Ok(self.gates[idx as usize].to_object(py))
                }
            }
        }
    }
}

fn circuit_kak(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    k_gate: &str,
    a_gate: &str,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut lam = lam;
    let mut theta = theta;
    let mut phi = phi;
    let mut circuit: Vec<(String, Vec<f64>)> = Vec::with_capacity(3);
    let mut atol = match atol {
        Some(atol) => atol,
        None => DEFAULT_ATOL,
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
            circuit.push((String::from(k_gate), vec![lam]));
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
        circuit.push((String::from(k_gate), vec![lam]));
    }
    circuit.push((String::from(a_gate), vec![theta]));
    phi = mod_2pi(phi, atol);
    if phi.abs() > atol {
        global_phase += phi / 2.;
        circuit.push((String::from(k_gate), vec![phi]));
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
        None => DEFAULT_ATOL,
    };
    let phi = mod_2pi(phi, atol);
    let lam = mod_2pi(lam, atol);
    if !simplify || theta.abs() > atol || phi.abs() > atol || lam.abs() > atol {
        circuit.push((String::from("u3"), vec![theta, phi, lam]));
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
        None => DEFAULT_ATOL,
    };
    if !simplify {
        atol = -1.0;
    }
    if theta.abs() < atol {
        let tot = mod_2pi(phi + lam, atol);
        if tot.abs() > atol {
            circuit.push((String::from("u1"), vec![tot]));
        }
    } else if (theta - PI / 2.).abs() < atol {
        circuit.push((
            String::from("u2"),
            vec![mod_2pi(phi, atol), mod_2pi(lam, atol)],
        ));
    } else {
        circuit.push((
            String::from("u3"),
            vec![theta, mod_2pi(phi, atol), mod_2pi(lam, atol)],
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
        None => DEFAULT_ATOL,
    };
    if !simplify {
        atol = -1.0;
    }
    let phi = mod_2pi(phi, atol);
    let lam = mod_2pi(lam, atol);
    if theta.abs() > atol || phi.abs() > atol || lam.abs() > atol {
        circuit.push((String::from("u"), vec![theta, phi, lam]));
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
        None => DEFAULT_ATOL,
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
        None => DEFAULT_ATOL,
    };
    if !simplify {
        atol = -1.0;
    }
    if theta.abs() < atol && phi.abs() < atol && lam.abs() < atol {
        return OneQubitGateSequence {
            gates: circuit,
            global_phase: phase,
        };
    }
    if (theta - PI).abs() > atol {
        circuit.push((
            String::from("r"),
            vec![theta - PI, mod_2pi(PI / 2. - lam, atol)],
        ));
    }
    circuit.push((
        String::from("r"),
        vec![PI, mod_2pi(0.5 * (phi - lam + PI), atol)],
    ));
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

#[pyfunction]
pub fn generate_circuit(
    target_basis: &str,
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<OneQubitGateSequence> {
    let res = match target_basis {
        "ZYZ" => circuit_kak(theta, phi, lam, phase, "rz", "ry", simplify, atol),
        "ZXZ" => circuit_kak(theta, phi, lam, phase, "rz", "rx", simplify, atol),
        "XZX" => circuit_kak(theta, phi, lam, phase, "rx", "rz", simplify, atol),
        "XYX" => circuit_kak(theta, phi, lam, phase, "rx", "ry", simplify, atol),
        "U3" => circuit_u3(theta, phi, lam, phase, simplify, atol),
        "U321" => circuit_u321(theta, phi, lam, phase, simplify, atol),
        "U" => circuit_u(theta, phi, lam, phase, simplify, atol),
        "PSX" => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => DEFAULT_ATOL,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((String::from("p"), vec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((String::from("sx"), Vec::new()));
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
        "ZSX" => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => DEFAULT_ATOL,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((String::from("rz"), vec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((String::from("sx"), Vec::new()));
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
        "U1X" => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => DEFAULT_ATOL,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((String::from("u1"), vec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.global_phase += PI / 4.;
                circuit.gates.push((String::from("rx"), vec![PI / 2.]));
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
        "ZSXX" => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => DEFAULT_ATOL,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((String::from("rz"), vec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((String::from("sx"), Vec::new()));
            };
            let fnxpi = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((String::from("x"), Vec::new()));
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
        "RR" => circuit_rr(theta, phi, lam, phase, simplify, atol),
        other => {
            return Err(PyTypeError::new_err(format!(
                "Invalid target basis: {other}"
            )))
        }
    };
    Ok(res)
}

#[inline]
fn angles_from_unitary(unitary: ArrayView2<Complex64>, target_basis: &str) -> [f64; 4] {
    match target_basis {
        "U321" => params_u3_inner(unitary),
        "U3" => params_u3_inner(unitary),
        "U" => params_u3_inner(unitary),
        "PSX" => params_u1x_inner(unitary),
        "ZSX" => params_u1x_inner(unitary),
        "ZSXX" => params_u1x_inner(unitary),
        "U1X" => params_u1x_inner(unitary),
        "RR" => params_zyz_inner(unitary),
        "ZYZ" => params_zyz_inner(unitary),
        "ZXZ" => params_zxz_inner(unitary),
        "XYX" => params_xyx_inner(unitary),
        "XZX" => params_xzx_inner(unitary),
        &_ => unreachable!(),
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
                .map(|x| 1. - err_map.get(&x.0).unwrap_or(&0.))
                .product();
            (1. - fidelity_product, circuit.gates.len())
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

fn compute_error(
    gates: &[(String, Vec<f64>)],
    error_map: Option<&OneQubitGateErrorMap>,
    qubit: usize,
) -> (f64, usize) {
    match error_map {
        Some(err_map) => {
            let num_gates = gates.len();
            let gate_fidelities: f64 = gates
                .iter()
                .map(|x| 1. - err_map.error_map[qubit].get(&x.0).unwrap_or(&0.))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (gates.len() as f64, gates.len()),
    }
}

#[pyfunction]
pub fn compute_error_one_qubit_sequence(
    circuit: &OneQubitGateSequence,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
) -> (f64, usize) {
    compute_error(&circuit.gates, error_map, qubit)
}

#[pyfunction]
pub fn compute_error_list(
    circuit: Vec<(String, Vec<f64>)>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
) -> (f64, usize) {
    compute_error(&circuit, error_map, qubit)
}

#[pyfunction]
#[pyo3(signature = (unitary, target_basis_list, qubit, error_map=None, simplify=true, atol=None))]
pub fn unitary_to_gate_sequence(
    unitary: PyReadonlyArray2<Complex64>,
    target_basis_list: Vec<&str>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<Option<OneQubitGateSequence>> {
    const VALID_BASES: [&str; 12] = [
        "U321", "U3", "U", "PSX", "ZSX", "ZSXX", "U1X", "RR", "ZYZ", "ZXZ", "XYX", "XZX",
    ];
    for basis in &target_basis_list {
        if !VALID_BASES.contains(basis) {
            return Err(PyTypeError::new_err(format!(
                "Invalid target basis {basis}"
            )));
        }
    }
    let unitary_mat = unitary.as_array();
    let best_result = target_basis_list
        .iter()
        .map(|target_basis| {
            let [theta, phi, lam, phase] = angles_from_unitary(unitary_mat, target_basis);
            generate_circuit(target_basis, theta, phi, lam, phase, simplify, atol).unwrap()
        })
        .min_by(|a, b| {
            let error_a = compare_error_fn(a, &error_map, qubit);
            let error_b = compare_error_fn(b, &error_map, qubit);
            error_a.partial_cmp(&error_b).unwrap_or(Ordering::Equal)
        });
    Ok(best_result)
}

#[inline]
fn det_one_qubit(mat: ArrayView2<Complex64>) -> Complex64 {
    mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]]
}

#[inline]
fn complex_phase(x: Complex64) -> f64 {
    x.im.atan2(x.re)
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
    let coeff: Complex64 = 1. / det_one_qubit(mat).sqrt();
    let phase = -complex_phase(coeff);
    let tmp_1_0 = (coeff * mat[[1, 0]]).abs();
    let tmp_0_0 = (coeff * mat[[0, 0]]).abs();
    let theta = 2. * tmp_1_0.atan2(tmp_0_0);
    let phiplambda2 = complex_phase(coeff * mat[[1, 1]]);
    let phimlambda2 = complex_phase(coeff * mat[[1, 0]]);
    let phi = phiplambda2 + phimlambda2;
    let lam = phiplambda2 - phimlambda2;
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
    let phase = (Complex64::new(0., -1.) * det.ln()).re / 2.;
    let sqrt_det = det.sqrt();
    let mat_zyz = arr2(&[
        [
            Complex64::new((umat[[0, 0]] / sqrt_det).re, (umat[[1, 0]] / sqrt_det).im),
            Complex64::new((umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
        ],
        [
            Complex64::new(-(umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
            Complex64::new((umat[[0, 0]] / sqrt_det).re, -(umat[[1, 0]] / sqrt_det).im),
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

#[pymodule]
pub fn euler_one_qubit_decomposer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(params_zyz))?;
    m.add_wrapped(wrap_pyfunction!(params_xyx))?;
    m.add_wrapped(wrap_pyfunction!(params_xzx))?;
    m.add_wrapped(wrap_pyfunction!(params_zxz))?;
    m.add_wrapped(wrap_pyfunction!(params_u3))?;
    m.add_wrapped(wrap_pyfunction!(params_u1x))?;
    m.add_wrapped(wrap_pyfunction!(generate_circuit))?;
    m.add_wrapped(wrap_pyfunction!(unitary_to_gate_sequence))?;
    m.add_wrapped(wrap_pyfunction!(compute_error_one_qubit_sequence))?;
    m.add_wrapped(wrap_pyfunction!(compute_error_list))?;
    m.add_class::<OneQubitGateSequence>()?;
    m.add_class::<OneQubitGateErrorMap>()?;
    Ok(())
}
