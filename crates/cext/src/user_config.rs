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

use ini::{Error as IniError, Ini};
use std::env;
use std::path::PathBuf;
use thiserror::Error;

use qiskit_transpiler::transpiler::OptimizationLevel;

#[derive(Debug, Error)]
pub(crate) enum UserConfigLoadError {
    #[error(transparent)]
    IniError(IniError),
    #[error("Invalid circuit_drawer type {0}")]
    InvalidCircuitDrawerType(String),
    #[error("Invalid state drawer type {0}")]
    InvalidStateDrawerType(String),
    #[error("Not a boolean value: {0}")]
    NotBoolean(String),
    #[error("Invalid Transpiler Optimization Level: {0}")]
    InvalidTranspilerOptimizationLevel(String),
    #[error("Invalid value for transpiler seed: {0}")]
    InvalidTranspilerSeed(String),
    #[error("Invalid value for num_processes: {0}")]
    InvalidNumProcs(String),
    #[error("Invalid value for min qpy version: {0}")]
    InvalidMinQpyVersion(String),
}

impl From<IniError> for UserConfigLoadError {
    fn from(value: IniError) -> Self {
        Self::IniError(value)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum CircuitDrawerMethod {
    Text,
    Mpl,
    Latex,
    LatexSource,
    Auto,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum StateDrawerMethod {
    Repr,
    Text,
    Latex,
    LatexSource,
    Qsphere,
    Hinton,
    Bloch,
}

#[derive(Debug, Default)]
pub(crate) struct ConfigurationFile {
    pub(crate) circuit_drawer: Option<CircuitDrawerMethod>,
    pub(crate) circuit_mpl_style: Option<String>,
    pub(crate) circuit_mpl_style_path: Option<String>,
    pub(crate) circuit_reverse_bits: Option<bool>,
    pub(crate) circuit_idle_wires: Option<bool>,
    pub(crate) transpile_optimization_level: Option<OptimizationLevel>,
    pub(crate) transpiler_seed: Option<u64>,
    pub(crate) parallel: Option<bool>,
    pub(crate) num_processes: Option<usize>,
    pub(crate) sabre_all_threads: Option<bool>,
    pub(crate) min_qpy_version: Option<u8>,
    pub(crate) state_drawer: Option<StateDrawerMethod>,
}

fn get_boolean(value: &str) -> Result<bool, UserConfigLoadError> {
    match value.to_lowercase().as_str() {
        "1" | "yes" | "true" | "on" => Ok(true),
        "0" | "no" | "false" | "off" => Ok(false),
        _ => Err(UserConfigLoadError::NotBoolean(value.to_string())),
    }
}

pub(crate) fn get_config_file() -> Result<Option<ConfigurationFile>, UserConfigLoadError> {
    if env::var("QISKIT_IGNORE_USER_SETTINGS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE"
    {
        return Ok(None);
    }
    let config_file_path = match env::var("QISKIT_SETTINGS") {
        Ok(x) => PathBuf::from(x),
        Err(_) => match env::home_dir() {
            None => return Ok(None),
            Some(home_dir) => home_dir.join(".qiskit").join("settings.conf"),
        },
    };
    if !config_file_path.exists() {
        return Ok(None);
    }
    let config_file = Ini::load_from_file(config_file_path)?;
    let mut out = ConfigurationFile::default();
    for (sec, prop) in &config_file {
        if sec != Some("default") {
            continue;
        }
        for (key, value) in prop.iter() {
            match key {
                "circuit_drawer" => match value {
                    "text" => out.circuit_drawer = Some(CircuitDrawerMethod::Text),
                    "mpl" => out.circuit_drawer = Some(CircuitDrawerMethod::Mpl),
                    "latex" => out.circuit_drawer = Some(CircuitDrawerMethod::Latex),
                    "latex_source" => out.circuit_drawer = Some(CircuitDrawerMethod::LatexSource),
                    "auto" => out.circuit_drawer = Some(CircuitDrawerMethod::Auto),
                    _ => {
                        return Err(UserConfigLoadError::InvalidCircuitDrawerType(
                            value.to_string(),
                        ));
                    }
                },
                "circuit_mpl_style" => out.circuit_mpl_style = Some(value.to_string()),
                "circuit_mpl_style_path" => out.circuit_mpl_style_path = Some(value.to_string()),
                "circuit_reverse_bits" => out.circuit_reverse_bits = Some(get_boolean(value)?),
                "circuit_idle_wires" => out.circuit_idle_wires = Some(get_boolean(value)?),
                "transpile_optimization_level" => match value {
                    "0" => out.transpile_optimization_level = Some(OptimizationLevel::Level0),
                    "1" => out.transpile_optimization_level = Some(OptimizationLevel::Level1),
                    "2" => out.transpile_optimization_level = Some(OptimizationLevel::Level2),
                    "3" => out.transpile_optimization_level = Some(OptimizationLevel::Level3),
                    _ => {
                        return Err(UserConfigLoadError::InvalidTranspilerOptimizationLevel(
                            value.to_string(),
                        ));
                    }
                },
                "transpiler_seed" => {
                    out.transpiler_seed = Some(value.parse().map_err(|_| {
                        UserConfigLoadError::InvalidTranspilerSeed(value.to_string())
                    })?)
                }
                "parallel" => out.parallel = Some(get_boolean(value)?),
                "num_processes" => {
                    out.num_processes = Some(
                        value
                            .parse()
                            .map_err(|_| UserConfigLoadError::InvalidNumProcs(value.to_string()))?,
                    )
                }
                "sabre_all_threads" => out.sabre_all_threads = Some(get_boolean(value)?),
                "min_qpy_version" => {
                    out.min_qpy_version = Some(value.parse().map_err(|_| {
                        UserConfigLoadError::InvalidMinQpyVersion(value.to_string())
                    })?)
                }
                "state_drawer" => match value {
                    "repr" => out.state_drawer = Some(StateDrawerMethod::Repr),
                    "text" => out.state_drawer = Some(StateDrawerMethod::Text),
                    "latex" => out.state_drawer = Some(StateDrawerMethod::Latex),
                    "latex_source" => out.state_drawer = Some(StateDrawerMethod::LatexSource),
                    "qsphere" => out.state_drawer = Some(StateDrawerMethod::Qsphere),
                    "hinton" => out.state_drawer = Some(StateDrawerMethod::Hinton),
                    "bloch" => out.state_drawer = Some(StateDrawerMethod::Bloch),
                    _ => {
                        return Err(UserConfigLoadError::InvalidStateDrawerType(
                            value.to_string(),
                        ));
                    }
                },
                _ => continue,
            }
        }
    }
    Ok(Some(out))
}
