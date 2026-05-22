// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::TranspilerError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{DelayUnit, OperationRef, StandardInstruction};

/// Run duration validation passes.
///
/// Args:
///     dag: DAG circuit to check instruction durations.
///     acquire_align: Integer number representing the minimum time resolution to
///         trigger acquisition instruction in units of dt.
///     pulse_align: Integer number representing the minimum time resolution to
///         trigger gate instruction in units of ``dt``.
/// Returns:
///     True if rescheduling is required, False otherwise.

#[pyfunction]
#[pyo3(signature=(dag, acquire_align, pulse_align))]
pub fn run_instruction_duration_check(
    _py: Python,
    dag: &DAGCircuit,
    acquire_align: u32,
    pulse_align: u32,
) -> PyResult<bool> {
    let num_stretches = dag.num_stretches();

    // Rescheduling is not necessary
    if (acquire_align == 1 && pulse_align == 1) || num_stretches != 0 {
        return Ok(false);
    }

    // Check delay durations
    for (_, packed_op) in dag.op_nodes(false) {
        if let OperationRef::StandardInstruction(StandardInstruction::Delay(handle)) =
            packed_op.op.view()
        {
            // The duration now lives in the delay arena; pull `(unit, value)`
            // directly rather than re-routing through `params_view`.
            let result: Result<bool, PyErr> = handle.with(|unit, data| {
                if unit != DelayUnit::DT {
                    return Err(TranspilerError::new_err(
                        "Delay duration must have dt unit for checking alignment.",
                    ));
                }
                let duration: u32 = match data {
                    qiskit_circuit::delay_arena::DelayDuration::Int(i) => u32::try_from(*i)
                        .map_err(|_| {
                            TranspilerError::new_err(
                                "The provided Delay duration is not in terms of dt.",
                            )
                        })?,
                    _ => {
                        return Err(TranspilerError::new_err(
                            "The provided Delay duration is not in terms of dt.",
                        ));
                    }
                };
                Ok(!(duration % acquire_align == 0 || duration % pulse_align == 0))
            });
            if result? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub fn instruction_duration_check_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_instruction_duration_check))?;
    Ok(())
}
