// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit};
use qiskit_circuit::dag_node::DAGOutNode;
use qiskit_circuit::operations::{
    DelayUnit, Param, StandardInstruction};
use qiskit_circuit::Qubit;
use smallvec::{smallvec};

#[pyfunction]
#[pyo3(name = "pad_delay")]
pub fn run_pad_delay(
    py: Python,
    dag: &mut DAGCircuit,
    qubit: ShareableQubit,
    // t_start: usize,
    // t_end: usize,
    t_start: f64,
    t_end: f64,
    fill_very_end: bool,
    next_node: &Bound<PyAny>, // Subclass of DAGNode is unknown
    // The Python equivalent this is replacing takes a prev_node 
    // but does nothing with it. Keep this variable? Rename to _prev_node?
    #[allow(unused_variables)]
    prev_node: &Bound<PyAny>, 
    property_set: &Bound<PyAny>,
) -> PyResult<()> {
    if !fill_very_end && (next_node.is_instance_of::<DAGOutNode>()) {
        return Ok(());
    }

    // NOTE: I think this is causing underflow?
    // Warnings in Tox test output such as:
    // ` UserWarning: Duration is rounded to 5952 [dt] = 1.322667e-06 [s] from 1.322676e-06 [s] `
    let time_interval = t_end - t_start;
    apply_scheduled_delay_op(
        py,
        dag,
        &t_start,
        &time_interval,
        // &(t_start as f64),
        // &(time_interval as f64),
        &qubit,
        &property_set,
    )?;

    Ok(())
}

/// Applies a delay operation to the DAGCircuit at the specified time interval.
/// It would make sense to replace this by a generic rust-native ``apply_scheduled_op``, once BOTH the ``PadDynamicalDecoupling`` and 
/// ``PadDelay`` passes (which inherit from ``BasePadding``) are oxidized, so it can be called for both passes...
fn apply_scheduled_delay_op(
    py: Python,
    dag: &mut DAGCircuit,
    // t_start: &usize,
    // time_interval: &usize,
    t_start: &f64,
    time_interval: &f64,
    qubit: &ShareableQubit,
    property_set: &Bound<PyAny>,
) -> PyResult<()> {

    // let delay_instr = StandardInstruction::Delay(map_delay_str_to_enum(
    //     &dag.get_internal_unit()
    //         .unwrap_or("dt".to_string())
    //         .to_string(),
    // ));

    let delay_instr = {
        let u = dag
            .get_internal_unit()
            .and_then(|s| DelayUnit::from_str(&s).map_err(PyErr::from))
            .unwrap_or(DelayUnit::DT);
        StandardInstruction::Delay(u)
    };

    
    // This seems to add a decimal point to dt when passing it back to python.
    // let params = Some(smallvec![Param::Float(*time_interval)]); // if passing a float directly
    
    // Why PyInt conversion? The internal representation of time 
    // added a decimal when passing a float back to python. The 
    // difference in internal representation breaks some tests such as:
    // `test.python.transpiler.test_context_aware_dd.TestContextAwareDD.test_collecting_diamond_with_initial`
    let py_dur: PyObject = (*time_interval as usize).into_pyobject(py).unwrap().into();
    let params = Some(smallvec![Param::Obj(py_dur)]);
    // let params = Some(smallvec![Param::Obj(time_interval.into_pyobject(py).unwrap().into())]); // python ints from usize
    // let params = Some(smallvec![Param::Float(*time_interval as f64)]); // casting usize to f64

    let new_node = dag.apply_operation_back(
        delay_instr.into(), 
        &[Qubit(qubit.index().unwrap() as u32)], 
        &[], 
        params, 
        None,
        #[cfg(feature = "cache_pygates")]
        None
    )?;

    let py_new_node = dag.get_node(py, new_node)?;
    let node_start_time_obj = property_set.get_item("node_start_time")?;
    let node_start_time_dict = node_start_time_obj.downcast::<PyDict>()?;
    
    // This seems to add a decimal point to the time interval when passing it back to python.
    // Again, the difference in internal representation breaks some tests
    // using f64
    // node_start_time_dict.set_item(&py_new_node, t_start)?;

    // using pyobject
    let py_start: PyObject = (*t_start as usize).into_pyobject(py).unwrap().into();
    node_start_time_dict.set_item(&py_new_node, py_start)?;

    Ok(())
}

pub fn pad_delay_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_pad_delay))?;
    Ok(())
}


// TODO use a function like this or the *new* Impl from_str in operations::DelayUnit ?
// fn map_delay_str_to_enum(delay_str: &str) -> DelayUnit {
//     match delay_str {
//         "ns" => DelayUnit::NS,
//         "ps" => DelayUnit::PS,
//         "us" => DelayUnit::US,
//         "ms" => DelayUnit::MS,
//         "s" => DelayUnit::S,
//         "dt" => DelayUnit::DT,
//         "expr" => DelayUnit::EXPR,
//         _ => unreachable!(),
//     }
// }