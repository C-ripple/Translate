use std::{
    env, fs,
    io::{Read, Write},
    path::PathBuf,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: ripple-codegen INPUT_REPORT.json OUTPUT.c".into());
    }
    let output = PathBuf::from(&args[1]);
    let input = PathBuf::from(&args[0]);
    if input.exists() && output.exists() && fs::canonicalize(&output)? == fs::canonicalize(&input)?
    {
        return Err("input and output must be different files".into());
    }
    #[cfg(unix)]
    if input.exists() && output.exists() {
        use std::os::unix::fs::MetadataExt;
        let a = fs::metadata(&input)?;
        let b = fs::metadata(&output)?;
        if a.dev() == b.dev() && a.ino() == b.ino() {
            return Err("input and output must not be hard-link aliases".into());
        }
    }
    if output.exists() {
        fs::remove_file(&output)?;
    }
    const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;
    let mut content = Vec::new();
    fs::File::open(input)?
        .take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut content)?;
    if content.len() as u64 > MAX_REPORT_BYTES {
        return Err(ripple_ir::Diagnostic::new(
            ripple_ir::Code::IrLimit,
            "input report exceeds 16 MiB",
        )
        .into());
    }
    let invalid_input =
        |message: String| ripple_ir::Diagnostic::new(ripple_ir::Code::InvalidInput, message);
    let report: serde_json::Value =
        serde_json::from_slice(&content).map_err(|e| invalid_input(e.to_string()))?;
    let module = serde_json::from_value(
        report
            .get("owned_ir")
            .ok_or_else(|| invalid_input("missing owned_ir".into()))?
            .clone(),
    )
    .map_err(|e| invalid_input(e.to_string()))?;
    let text = ripple_codegen::emit(ripple_ir::verify(module)?)?;
    let temporary = output.with_file_name(format!(".ripple-codegen-{}.tmp", std::process::id()));
    let published = (|| -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary, &output)
    })();
    if published.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    published?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        let diagnostic = error
            .downcast_ref::<ripple_ir::Diagnostic>()
            .cloned()
            .unwrap_or_else(|| {
                ripple_ir::Diagnostic::new(ripple_ir::Code::Pipeline, error.to_string())
            });
        eprintln!(
            "RIPPLE_DIAGNOSTIC: {}",
            serde_json::to_string(&diagnostic).expect("diagnostic serialization")
        );
        eprintln!("{diagnostic}");
        std::process::exit(1);
    }
}
