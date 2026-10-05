use std::process::Command;

fn main() {
    let rustc = std::env::var_os("RUSTC").expect("Cargo must provide RUSTC");
    let output = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("query rustc sysroot");
    assert!(output.status.success(), "rustc sysroot query failed");
    let sysroot = String::from_utf8(output.stdout).expect("UTF-8 sysroot");
    // rustc_driver is a dylib outside the usual runtime library search path.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
    println!("cargo:rerun-if-env-changed=RUSTC");
}
