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

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    str::FromStr,
};

extern crate cbindgen;

/// This function generates the C header for Qiskit from the qiskit-cext crate.
fn generate_qiskit_header() {
    // Trigger this script if the header was changed/removed.
    #![allow(clippy::print_stdout)]
    println!("cargo:rerun-if-changed=../../target/qiskit.h");

    // Pull the config from the cbindgen.toml file.
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap();

    // Ensure target path exists and then set the full filename of qiskit.h.
    let mut path = ::std::path::PathBuf::from_str("../../target/").unwrap();
    ::std::fs::create_dir_all(&path).expect("Could not create target directory.");
    // We use a temporary file for cbindgen, which we post-process to fix renamings we
    // don't want (like PyObject being renamed for QkPyObject).
    let mut tmp_header = path.clone();
    tmp_header.push("__tmp_qiskit.h");

    path.push("qiskit.h");

    // Build the header.
    cbindgen::Builder::new()
        .with_crate(".")
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings.")
        .write_to_file(&tmp_header);

    // Undo cbindgen's renaming of QkPyObject to PyObject
    let final_file = File::create(&path).expect("The qiskit.h path should exist.");
    let mut final_buffer = BufWriter::new(final_file);
    let tmp_file = File::open(&tmp_header).expect("The tmp path should exist.");
    let tmp_buffer = BufReader::new(tmp_file);

    // we currently only have a single target-replacement pair, this could be promoted to a
    // HashMap if we have more than that
    let target = "QkPyObject";
    let replacement = "PyObject";

    // Check each line and replace the target words
    for line in tmp_buffer.lines() {
        let mut line = line.expect("The tmp header file should be readable");
        if line.contains(target) {
            line = line.replace(target, replacement);
        }
        writeln!(final_buffer, "{}", line).expect("Failed writing to qiskit.h");
    }
    final_buffer.flush().expect("Failed writing to qiskit.h");
    ::std::fs::remove_file(tmp_header).expect("Failed deleting tmp header.");
}

fn main() {
    generate_qiskit_header();
}
