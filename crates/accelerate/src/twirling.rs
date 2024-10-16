// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use std::f64::consts::PI;

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::StandardGate::{IGate, XGate, YGate, ZGate};
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_circuit::Qubit;

use crate::QiskitError;

static ECR_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([IGate, ZGate, ZGate, YGate], 0.),
    ([IGate, XGate, IGate, XGate], 0.),
    ([IGate, YGate, ZGate, ZGate], PI),
    ([IGate, IGate, IGate, IGate], 0.),
    ([ZGate, XGate, ZGate, XGate], PI),
    ([ZGate, YGate, IGate, ZGate], 0.),
    ([ZGate, IGate, ZGate, IGate], PI),
    ([ZGate, ZGate, IGate, YGate], PI),
    ([XGate, YGate, XGate, YGate], 0.),
    ([XGate, IGate, YGate, XGate], PI),
    ([XGate, ZGate, XGate, ZGate], 0.),
    ([XGate, XGate, YGate, IGate], PI),
    ([YGate, IGate, XGate, XGate], PI),
    ([YGate, ZGate, YGate, ZGate], PI),
    ([YGate, XGate, XGate, IGate], PI),
    ([YGate, YGate, YGate, YGate], PI),
];

static CX_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([IGate, ZGate, ZGate, ZGate], 0.),
    ([IGate, XGate, IGate, XGate], 0.),
    ([IGate, YGate, ZGate, YGate], 0.),
    ([IGate, IGate, IGate, IGate], 0.),
    ([ZGate, XGate, ZGate, XGate], 0.),
    ([ZGate, YGate, IGate, YGate], 0.),
    ([ZGate, IGate, ZGate, IGate], 0.),
    ([ZGate, ZGate, IGate, ZGate], 0.),
    ([XGate, YGate, YGate, ZGate], 0.),
    ([XGate, IGate, XGate, XGate], 0.),
    ([XGate, ZGate, YGate, YGate], PI),
    ([XGate, XGate, XGate, IGate], 0.),
    ([YGate, IGate, YGate, XGate], 0.),
    ([YGate, ZGate, XGate, YGate], 0.),
    ([YGate, XGate, YGate, IGate], 0.),
    ([YGate, YGate, XGate, ZGate], PI),
];

static CZ_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([IGate, ZGate, IGate, ZGate], 0.),
    ([IGate, XGate, ZGate, XGate], 0.),
    ([IGate, YGate, ZGate, YGate], 0.),
    ([IGate, IGate, IGate, IGate], 0.),
    ([ZGate, XGate, IGate, XGate], 0.),
    ([ZGate, YGate, IGate, YGate], 0.),
    ([ZGate, IGate, ZGate, IGate], 0.),
    ([ZGate, ZGate, ZGate, ZGate], 0.),
    ([XGate, YGate, YGate, XGate], PI),
    ([XGate, IGate, XGate, ZGate], 0.),
    ([XGate, ZGate, XGate, IGate], 0.),
    ([XGate, XGate, YGate, YGate], 0.),
    ([YGate, IGate, YGate, ZGate], 0.),
    ([YGate, ZGate, YGate, IGate], 0.),
    ([YGate, XGate, XGate, YGate], PI),
    ([YGate, YGate, XGate, XGate], 0.),
];

static ISWAP_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([IGate, ZGate, ZGate, IGate], 0.),
    ([IGate, XGate, YGate, ZGate], 0.),
    ([IGate, YGate, XGate, ZGate], PI),
    ([IGate, IGate, IGate, IGate], 0.),
    ([ZGate, XGate, YGate, IGate], 0.),
    ([ZGate, YGate, XGate, IGate], PI),
    ([ZGate, IGate, IGate, ZGate], 0.),
    ([ZGate, ZGate, ZGate, ZGate], 0.),
    ([XGate, YGate, YGate, XGate], 0.),
    ([XGate, IGate, ZGate, YGate], 0.),
    ([XGate, ZGate, IGate, YGate], 0.),
    ([XGate, XGate, XGate, XGate], 0.),
    ([YGate, IGate, ZGate, XGate], PI),
    ([YGate, ZGate, IGate, XGate], PI),
    ([YGate, XGate, XGate, YGate], 0.),
    ([YGate, YGate, YGate, YGate], 0.),
];

#[pyfunction]
#[pyo3(signature=(circ, twirled_gate, seed=None, num_twirls=1))]
pub fn twirl_circuit(
    py: Python,
    circ: &CircuitData,
    twirled_gate: StandardGate,
    seed: Option<u64>,
    num_twirls: usize,
) -> PyResult<Vec<CircuitData>> {
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let twirl_set: &[([StandardGate; 4], f64)] = match twirled_gate {
        StandardGate::CXGate => &CX_TWIRL_SET,
        StandardGate::CZGate => &CZ_TWIRL_SET,
        StandardGate::ECRGate => &ECR_TWIRL_SET,
        StandardGate::ISwapGate => &ISWAP_TWIRL_SET,
        _ => {
            return Err(QiskitError::new_err(
                "Provided gate to twirl is not currently supported you can only use CX, CZ, ECR or iSwap.",
            ))
        }
    };
    let generate_twirled_circuit = |rng: &mut Pcg64Mcg| {
        let mut out_circ = CircuitData::clone_empty_from(circ, None);

        for inst in circ.data() {
            match inst.op.view() {
                OperationRef::Standard(gate) => {
                    if gate == twirled_gate {
                        let qubits: Vec<Qubit> = out_circ.get_qargs(inst.qubits).to_vec();
                        let (twirl, twirl_phase) = twirl_set.choose(rng).unwrap();
                        out_circ.push_standard_gate(twirl[0], &[], &[qubits[0]])?;
                        out_circ.push_standard_gate(twirl[1], &[], &[qubits[1]])?;
                        out_circ.push(py, inst.clone())?;
                        out_circ.push_standard_gate(twirl[2], &[], &[qubits[0]])?;
                        out_circ.push_standard_gate(twirl[3], &[], &[qubits[1]])?;
                        if *twirl_phase != 0. {
                            out_circ.add_global_phase(py, &Param::Float(*twirl_phase))?;
                        }
                    } else {
                        out_circ.push(py, inst.clone())?;
                    }
                }
                _ => {
                    out_circ.push(py, inst.clone())?;
                }
            }
        }

        Ok(out_circ)
    };
    if num_twirls <= 4 {
        (0..num_twirls)
            .map(|_| generate_twirled_circuit(&mut rng))
            .collect()
    } else {
        let seed_vec: Vec<u64> = rand::distributions::Standard
            .sample_iter(&mut rng)
            .take(num_twirls)
            .collect();
        // TODO: Use into_par_iter() after CircuitData is made threadsafe
        // (see https://github.com/Qiskit/qiskit/issues/13219)
        seed_vec
            .into_iter()
            .map(|seed| {
                let mut inner_rng = Pcg64Mcg::seed_from_u64(seed);
                generate_twirled_circuit(&mut inner_rng)
            })
            .collect()
    }
}

pub fn twirling(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(twirl_circuit))?;
    Ok(())
}
