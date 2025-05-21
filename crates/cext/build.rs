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

use std::fs;
use std::str::FromStr;

extern crate cbindgen;

/// This function generates the version.h file from the VERSION.txt file information.
fn generate_version_header() {
    //Reading the version string from the VERSION.txt file.
    let version_file_path = "../../qiskit/VERSION.txt";
    let version_h_file_path = "./version.h";
    let qiskit_version = fs::read_to_string(version_file_path)
        .expect("Failed to read VERSION.txt file!")
        .trim()
        .to_string();

    // Obtain the major, minor and patch version numbers.
    let mut part = qiskit_version.split('.');
    let major = part.next().unwrap_or("0");
    let minor = part.next().unwrap_or("0");
    let patch = part.next().unwrap_or("0");

    // Read the existing version.h content
    let mut version_h_content = fs::read_to_string(version_h_file_path).unwrap();

    // Define the regex patterns to match the version lines in version.h
    use regex::Regex;
    let re_major = Regex::new(r#"#define QISKIT_VERSION_MAJOR.*"#).unwrap();
    let re_minor = Regex::new(r#"#define QISKIT_VERSION_MINOR.*"#).unwrap();
    let re_patch = Regex::new(r#"#define QISKIT_VERSION_PATCH.*"#).unwrap();

    // Replace the version lines with the new version numbers and write to version.h
    version_h_content = re_major
        .replace_all(
            &version_h_content,
            format!("#define QISKIT_VERSION_MAJOR {}", major),
        )
        .to_string();
    version_h_content = re_minor
        .replace_all(
            &version_h_content,
            format!("#define QISKIT_VERSION_MINOR {}", minor),
        )
        .to_string();
    version_h_content = re_patch
        .replace_all(
            &version_h_content,
            format!("#define QISKIT_VERSION_PATCH {}", patch),
        )
        .to_string();

    fs::write(version_h_file_path, version_h_content).expect("Failed to write version.h file!");
}

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
    path.push("qiskit.h");

    // Build the header.
    cbindgen::Builder::new()
        .with_crate(".")
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings.")
        .write_to_file(path);
}

fn main() {
    generate_version_header();
    generate_qiskit_header();
}
