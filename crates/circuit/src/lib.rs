// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod bit;
pub mod bit_locator;
pub mod circuit_data;
pub mod circuit_instruction;
pub mod classical;
pub mod converters;
pub mod dag_circuit;
pub mod dag_node;
mod dot_utils;
pub mod duration;
pub mod error;
pub mod gate_matrix;
pub mod imports;
pub mod interner;
pub mod nlayout;
pub mod object_registry;
pub mod operations;
pub mod packed_instruction;
pub mod parameter_table;
pub mod register_data;
pub mod slice;
pub mod util;

pub mod rustworkx_core_vnext;

use pyo3::prelude::*;
use pyo3::types::{PySequence, PyTuple};
use pyo3::PyTypeInfo;

#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq, FromPyObject)]
pub struct Qubit(pub u32);

#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq, FromPyObject)]
pub struct Clbit(pub u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Var(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Stretch(u32);

pub use nlayout::PhysicalQubit;
pub use nlayout::VirtualQubit;

macro_rules! impl_circuit_identifier {
    ($type:ident) => {
        impl $type {
            /// Construct a new identifier from a usize, if you have a u32 you can
            /// construct one directly via [$type()]. This will panic if the `usize`
            /// index exceeds `u32::MAX`.
            #[inline(always)]
            pub fn new(index: usize) -> Self {
                $type(index.try_into().unwrap_or_else(|_| {
                    panic!(
                        "Index value '{}' exceeds the maximum identifier width!",
                        index
                    )
                }))
            }

            /// Convert to a usize.
            #[inline(always)]
            pub fn index(&self) -> usize {
                self.0 as usize
            }
        }

        impl From<u32> for $type {
            fn from(value: u32) -> Self {
                $type(value)
            }
        }

        impl From<$type> for u32 {
            fn from(value: $type) -> Self {
                value.0
            }
        }
    };
}

impl_circuit_identifier!(Qubit);
impl_circuit_identifier!(Clbit);
impl_circuit_identifier!(Var);
impl_circuit_identifier!(Stretch);

pub struct TupleLikeArg<'py> {
    value: Bound<'py, PyTuple>,
}

impl<'py> FromPyObject<'py> for TupleLikeArg<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = match ob.downcast::<PySequence>() {
            Ok(seq) => seq.to_tuple()?,
            Err(_) => PyTuple::new(
                ob.py(),
                ob.try_iter()?
                    .map(|o| Ok(o?.unbind()))
                    .collect::<PyResult<Vec<PyObject>>>()?,
            )?,
        };
        Ok(TupleLikeArg { value })
    }
}

/// Implement `IntoPyObject` for the reference to a struct or enum declared as `#[pyclass]` that is
/// also `Copy`.
///
/// For example:
/// ```
/// #[derive(Clone, Copy)]
/// #[pyclass(frozen)]
/// struct MyStruct(u32);
///
/// impl_intopyobject_for_copy_pyclass!(MyStruct);
/// ```
///
/// The `pyclass` attribute macro already ensures that `IntoPyObject` is implemented for `MyStruct`,
/// but it doesn't implement it for `&MyStruct` - for non-copy structs, the implementation of that
/// is not obvious and may be surprising to users if it existed.  If the struct is `Copy`, though,
/// it's explicitly "free" to make new copies and convert them, so we can do that and delegate.
///
/// Usually this doesn't matter much to code authors, but it can help a lot when dealing with
/// references nested in ad-hoc structures, like `(&T1, &T2)`.
#[macro_export]
macro_rules! impl_intopyobject_for_copy_pyclass {
    ($ty:ty) => {
        impl<'py> ::pyo3::conversion::IntoPyObject<'py> for &$ty {
            type Target = <$ty as ::pyo3::conversion::IntoPyObject<'py>>::Target;
            type Output = <$ty as ::pyo3::conversion::IntoPyObject<'py>>::Output;
            type Error = <$ty as ::pyo3::conversion::IntoPyObject<'py>>::Error;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                (*self).into_pyobject(py)
            }
        }
    };
}

pub fn circuit(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<bit::PyBit>()?;
    m.add_class::<bit::PyClbit>()?;
    m.add_class::<bit::PyQubit>()?;
    m.add_class::<bit::PyAncillaQubit>()?;
    m.add_class::<bit::PyRegister>()?;
    m.add_class::<bit::PyClassicalRegister>()?;
    m.add_class::<bit::PyQuantumRegister>()?;
    m.add_class::<bit::PyAncillaRegister>()?;

    // We need to explicitly add the auto-generated Python subclasses of Duration
    // to the module so that pickle can find them during deserialization.
    m.add_class::<duration::Duration>()?;
    m.add(
        "Duration_ns",
        duration::Duration::type_object(m.py()).getattr("ns")?,
    )?;
    m.add(
        "Duration_us",
        duration::Duration::type_object(m.py()).getattr("us")?,
    )?;
    m.add(
        "Duration_ms",
        duration::Duration::type_object(m.py()).getattr("ms")?,
    )?;
    m.add(
        "Duration_s",
        duration::Duration::type_object(m.py()).getattr("s")?,
    )?;
    m.add(
        "Duration_dt",
        duration::Duration::type_object(m.py()).getattr("dt")?,
    )?;

    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    m.add_class::<dag_circuit::DAGCircuit>()?;
    m.add_class::<dag_node::DAGNode>()?;
    m.add_class::<dag_node::DAGInNode>()?;
    m.add_class::<dag_node::DAGOutNode>()?;
    m.add_class::<dag_node::DAGOpNode>()?;
    m.add_class::<dag_circuit::PyBitLocations>()?;
    m.add_class::<operations::StandardGate>()?;
    m.add_class::<operations::StandardInstructionType>()?;

    let classical_mod = PyModule::new(m.py(), "classical")?;
    classical::register_python(&classical_mod)?;
    m.add_submodule(&classical_mod)?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_qubit_create() {
        let expected = Qubit(12345);
        let val = 12345_usize;
        let result = Qubit::new(val);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_qubit_index_too_large() {
        let val = u32::MAX as usize + 42;
        Qubit::new(val);
    }

    #[test]
    fn test_clbit_create() {
        let expected = Clbit(12345);
        let val = 12345_usize;
        let result = Clbit::new(val);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_clbit_index_too_large() {
        let val = u32::MAX as usize + 42;
        Clbit::new(val);
    }

    #[test]
    fn test_qubit_index() {
        let qubit = Qubit(123456789);
        let expected = 123456789_usize;
        let result = qubit.index();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_clbit_index() {
        let clbit = Clbit(1234542);
        let expected = 1234542_usize;
        let result = clbit.index();
        assert_eq!(result, expected);
    }
}
