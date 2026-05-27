// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt;

use anyhow::anyhow;
use indexmap::IndexMap;
use qiskit_cext_vtable::ExportedFunction;

/// The minimal semver components extracted from a version that might also have suffixes.
#[derive(Clone, Debug)]
pub struct Version {
    major: usize,
    minor: usize,
    patch: usize,
    suffix: Option<String>,
}
impl Version {
    /// Parse a version out of a `<major>.<minor>.<patch>(-<suffix>)?` form.
    fn try_parse(val: &str) -> anyhow::Result<Version> {
        let (val, suffix) = match val.split_once("-") {
            Some((val, suffix)) => (val, Some(suffix.to_owned())),
            None => (val, None),
        };
        let mut parts = val.split(".");
        let mut part = || -> anyhow::Result<usize> {
            Ok(parts
                .next()
                .ok_or_else(|| anyhow!("not enough version parts"))?
                .parse()?)
        };
        Ok(Version {
            major: part()?,
            minor: part()?,
            patch: part()?,
            suffix,
        })
    }
}
impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(suffix) = self.suffix.as_ref() {
            write!(f, "-{suffix}")?;
        }
        Ok(())
    }
}

/// Complete set of exported slots and version for comparison.
pub struct SlotsLists {
    pub api_version: Version,
    /// Association of names to concrete vtables.
    pub slots: IndexMap<String, SlotsList>,
}
impl SlotsLists {
    /// The slots lists that are defined in the depended-on `qiskit_cext_vtable`.
    pub fn ours() -> Self {
        let names = |funcs: Vec<Option<ExportedFunction>>| {
            SlotsList(
                funcs
                    .into_iter()
                    .map(|f| f.map(|f| f.name.to_owned()))
                    .collect(),
            )
        };
        Self {
            api_version: Version::try_parse(env!("CARGO_PKG_VERSION"))
                .expect("our version should be valid"),
            slots: [
                ("circuit", &qiskit_cext_vtable::FUNCTIONS_CIRCUIT),
                ("transpile", &qiskit_cext_vtable::FUNCTIONS_TRANSPILE),
                ("qi", &qiskit_cext_vtable::FUNCTIONS_QI),
            ]
            .iter()
            .map(|(name, funcs)| (String::from(*name), names(funcs.slots())))
            .collect(),
        }
    }
}
impl fmt::Display for SlotsLists {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "__version__ = \"{}\"", &self.api_version)?;
        for (name, slots) in self.slots.iter() {
            writeln!(f, "{name} = {slots}")?;
        }
        Ok(())
    }
}

/// An individual concrete vtable, but written in terms of the function name instead of a pointer.
pub struct SlotsList(Vec<Option<String>>);
impl SlotsList {
    /// Iterate over the slot indices and the name stored there in order.
    pub fn iter_names(&self) -> impl Iterator<Item = (usize, &str)> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, fname)| fname.as_deref().map(|fname| (i, fname)))
    }
}
impl fmt::Display for SlotsList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // This is meant to be easy both to parse back in, and to be read by humans.  As written,
        // this `impl` also produces a valid Python list of strings that can be read by
        // `ast.literal_eval`.
        if self.0.is_empty() {
            write!(f, "[]")
        } else {
            writeln!(f, "[")?;
            for slot in self.0.iter() {
                writeln!(f, "    \"{}\",", slot.as_deref().unwrap_or_default())?;
            }
            write!(f, "]")
        }
    }
}
