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

use anyhow::{Context, anyhow};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub static GENERATED_FILE: &str = "qiskit.h";
pub static PYTHON_BINDING_FEATURE: &str = "python_binding";
pub static PYTHON_BINDING_DEFINE: &str = "QISKIT_C_PYTHON_INTERFACE";

pub static INCLUDE_GUARD: &str = "QISKIT_H";
/// Crates that contain definitions of objects that are exposed through the C API.
pub static QISKIT_PUBLIC_API_CRATES: &[&str] =
    &["qiskit-quantum-info", "qiskit-circuit", "qiskit-transpiler"];

pub static EXPORT_PREFIX: &str = "Qk";
pub static EXPORT_RENAME: &[(&str, &str)] = &[
    ("CBlocksMode", "BlocksMode"),
    ("CDagNeighbors", "DagNeighbors"),
    ("CDagNodeType", "DagNodeType"),
    ("CDelayUnit", "DelayUnit"),
    ("CInstruction", "CircuitInstruction"),
    ("CInstructionProperties", "InstructionProperties"),
    ("CNeighbors", "Neighbors"),
    ("COperationKind", "OperationKind"),
    ("CSparseTerm", "ObsTerm"),
    ("CTargetOp", "TargetOp"),
    ("CVarsMode", "VarsMode"),
    ("CircuitData", "Circuit"),
    ("DAGCircuit", "Dag"),
    ("SparseObservable", "Obs"),
    ("StandardGate", "Gate"),
];
pub static EXPORT_VERBATIM: &[&str] = &["PyObject"];

// Defined in `qiskit/attributes.h`.
pub static FN_DEPRECATED: &str = "Qk_DEPRECATED_FN";
pub static FN_DEPRECATED_WITH_NOTE: &str = "Qk_DEPRECATED_FN_NOTE({})";

/// Mapping of Rust `#[cfg(feature = <key>)]` keys to C `#ifdef <value>` values.
pub static CFG_FEATURE_DEFINES: &[(&str, &str)] =
    &[(PYTHON_BINDING_FEATURE, PYTHON_BINDING_DEFINE)];

fn guarded_python_import(guard: &str) -> String {
    format!(
        "\
#ifdef {guard}
#include <Python.h>
#endif"
    )
}

#[inline]
fn to_vec_string(slice: &[&str]) -> Vec<String> {
    slice.iter().map(|s| String::from(*s)).collect()
}

#[inline]
fn manual_include_dir() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "include"]
        .iter()
        .collect::<PathBuf>()
}
/// Get the manually written include files, relative to the result of [`manual_include_dir`].
fn manual_include_files() -> anyhow::Result<Vec<PathBuf>> {
    fn recurse(
        relative_path: &Path,
        base_dir: &Path,
        mut out: Vec<PathBuf>,
    ) -> anyhow::Result<Vec<PathBuf>> {
        for entry in base_dir.read_dir()? {
            let entry = entry?;
            let name = entry.file_name();
            let path = entry.path();
            if entry.metadata()?.is_dir() {
                out = recurse(&relative_path.join(&name), &base_dir.join(&name), out)?;
            } else {
                // Only propagate files that look like C/C++ header files.
                match path.extension().and_then(|s| s.to_str()) {
                    Some("h") | Some("hpp") => out.push(relative_path.join(&name)),
                    Some(_) | None => (),
                }
            }
        }
        Ok(out)
    }
    recurse(&PathBuf::new(), &manual_include_dir(), Vec::new())
}

/// Get the Qiskit configuration
fn get_config() -> anyhow::Result<cbindgen::Config> {
    let enumeration = cbindgen::EnumConfig {
        prefix_with_name: true,
        ..Default::default()
    };
    // For the export configuration, we need certain names ("EXPORT_VERBATIM") to stay as they are,
    // so we instead set `renaming_overrides_prefixing: true`, and manually propagate the prefix
    // onto all the renames, so we can re-use the "renames" mapping to force the verbatim exports
    // too.
    #[expect(clippy::disallowed_types)] // used in the cbindgen API.
    let mut rename: std::collections::HashMap<_, _> = EXPORT_RENAME
        .iter()
        .map(|&(k, v)| (String::from(k), format!("{EXPORT_PREFIX}{v}")))
        .collect();
    rename.extend(
        EXPORT_VERBATIM
            .iter()
            .map(|&k| (String::from(k), String::from(k))),
    );
    let export = cbindgen::ExportConfig {
        prefix: Some(EXPORT_PREFIX.into()),
        rename,
        renaming_overrides_prefixing: true,
        ..Default::default()
    };
    let function = cbindgen::FunctionConfig {
        deprecated: Some(FN_DEPRECATED.into()),
        deprecated_with_note: Some(FN_DEPRECATED_WITH_NOTE.into()),
        ..Default::default()
    };
    let parse = cbindgen::ParseConfig {
        parse_deps: true,
        include: Some(to_vec_string(QISKIT_PUBLIC_API_CRATES)),
        ..Default::default()
    };
    let defines = CFG_FEATURE_DEFINES
        .iter()
        .map(|&(cfg, def)| (format!("feature = {cfg}"), String::from(def)))
        .collect();
    // We always use `/` as the path separator to be portable.
    let includes = manual_include_files()?
        .into_iter()
        .map(|buf| {
            buf.iter()
                .try_fold(String::new(), |mut acc, part| -> anyhow::Result<_> {
                    let part = part
                        .to_os_string()
                        .into_string()
                        .map_err(|e| anyhow!(e.to_string_lossy().into_owned()))
                        .context("path could not be converted to UTF-8")?;
                    if !acc.is_empty() {
                        acc.push('/');
                    }
                    acc.push_str(&part);
                    Ok(acc)
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(cbindgen::Config {
        // `Python.h` is required to be the first file included because it reserves the right to
        // define preprocessor macros that affect standard-library includes.  This causes it to be
        // ahead of our include guard, but `Python.h` has its own, so we should be fine.
        header: Some(guarded_python_import(PYTHON_BINDING_DEFINE)),
        language: cbindgen::Language::C,
        include_version: true,
        include_guard: Some(INCLUDE_GUARD.into()),
        includes,
        style: cbindgen::Style::Type,
        cpp_compat: true,
        usize_is_size_t: true,
        defines,
        enumeration,
        export,
        function,
        parse,
        ..Default::default()
    })
}

/// Generate the cbindgen bindings object for the C-extensions crate.
pub fn generate_bindings(cext_path: impl AsRef<Path>) -> anyhow::Result<cbindgen::Bindings> {
    cbindgen::Builder::new()
        .with_crate(cext_path)
        .with_config(get_config()?)
        .generate()
        .map_err(|e| e.into())
}

/// Install the complete stand-alone C include path into the given directory.
pub fn install_c_headers(
    bindings: &cbindgen::Bindings,
    install_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let install_path = install_path.as_ref();
    fs::create_dir_all(install_path)?;
    let mut buf = Vec::<u8>::new();
    bindings.write(&mut buf);
    fs::File::create(install_path.join(GENERATED_FILE))?.write_all(&buf)?;
    let manual_path = manual_include_dir();
    for file in manual_include_files()? {
        if let Some(parent) = file.parent() {
            fs::create_dir_all(install_path.join(parent))?;
        }
        fs::copy(manual_path.join(&file), install_path.join(&file))?;
    }
    Ok(())
}
