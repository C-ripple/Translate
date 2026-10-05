use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ripple-publication-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fails(input: &PathBuf, output: &PathBuf) {
    let result = Command::new(env!("CARGO_BIN_EXE_ripple-codegen"))
        .arg(input)
        .arg(output)
        .output()
        .unwrap();
    assert!(!result.status.success());
}

#[test]
fn identical_path_preserves_input() {
    let dir = Directory::new();
    let input = dir.0.join("input.json");
    fs::write(&input, "important input").unwrap();
    fails(&input, &input);
    assert_eq!(fs::read_to_string(input).unwrap(), "important input");
}

#[test]
fn hard_link_alias_preserves_input() {
    let dir = Directory::new();
    let input = dir.0.join("input.json");
    let output = dir.0.join("output.c");
    fs::write(&input, "important input").unwrap();
    fs::hard_link(&input, &output).unwrap();
    fails(&input, &output);
    assert_eq!(fs::read_to_string(input).unwrap(), "important input");
}

#[test]
fn invalid_or_missing_input_invalidates_prior_output() {
    let dir = Directory::new();
    let input = dir.0.join("input.json");
    let output = dir.0.join("output.c");
    for exists in [false, true] {
        if exists {
            fs::write(&input, "not json").unwrap();
        }
        fs::write(&output, "stale successful output").unwrap();
        fails(&input, &output);
        assert!(!output.exists());
    }
}

#[test]
fn oversized_input_invalidates_output_with_limit_code() {
    let dir = Directory::new();
    let input = dir.0.join("input.json");
    let output = dir.0.join("output.c");
    fs::File::create(&input)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    fs::write(&output, "stale output").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ripple-codegen"))
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("RIPPLE-I006"));
}

#[test]
fn unknown_ir_fields_are_rejected() {
    let module = serde_json::json!({
        "schema_version": 1, "target": "hexagon-unknown-none-elf",
        "pointer_bits": 32, "functions": [], "silently_ignored": true
    });
    assert!(serde_json::from_value::<ripple_ir::Module>(module).is_err());
}

#[test]
fn nested_unknown_field_is_rejected_by_cli_without_stale_output() {
    let dir = Directory::new();
    let input = dir.0.join("input.json");
    let output = dir.0.join("output.c");
    fs::write(
        &input,
        serde_json::json!({"owned_ir": {
            "schema_version": 1, "target": "hexagon-unknown-none-elf", "pointer_bits": 32,
            "functions": [{"name": "test", "source": "test.rs:1", "locals": ["Unit"],
                "argument_count": 0, "blocks": [], "unknown_effect": true}]
        }})
        .to_string(),
    )
    .unwrap();
    fs::write(&output, "stale").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ripple-codegen"))
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("RIPPLE-D001"));
    assert!(stderr.contains("unknown_effect"));
}
