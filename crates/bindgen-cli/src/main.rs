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

mod abi;
mod lint;

use std::path::PathBuf;

use anyhow::anyhow;
use clap::{Parser, Subcommand};

use crate::abi::SlotsLists;

/// Toolkit for working with and installing the C header files.
#[derive(Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}
#[derive(Subcommand, Debug)]
enum Command {
    /// Install the header files into a given directory.
    Install {
        /// Path to the `cext` sources to generate headers for.
        #[arg(short, long)]
        cext_path: PathBuf,
        /// Where to install the header files to.
        #[arg(short, long)]
        output_path: PathBuf,
    },
    /// Check for correctness between the slots tables and the list of exported functions for the
    /// current version of Qiskit.
    LintSlots {
        /// Path to the `cext` sources to generate headers for.
        #[arg(short, long)]
        cext_path: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match &args.command {
        Command::Install {
            cext_path,
            output_path,
        } => {
            let mut bindings = qiskit_bindgen::generate_bindings(cext_path)?;
            qiskit_bindgen::install_c_headers(&mut bindings, output_path)?;
            Ok(())
        }
        Command::LintSlots { cext_path } => {
            let bindings = qiskit_bindgen::generate_bindings(cext_path)?;
            lint::lint(&bindings, &SlotsLists::ours())?.map_err(|fails| anyhow!(fails.explain()))
        }
    }
}
