fn main() {
    // Cargo otherwise has no reason to rerun rustc when the analysis entry or
    // report destination changes. These are inputs to the workspace wrapper.
    println!("cargo:rerun-if-env-changed=RIPPLE_PROBE_ENTRY");
    println!("cargo:rerun-if-env-changed=RIPPLE_PROBE_REPORT");
}
