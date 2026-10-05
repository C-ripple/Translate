#!/usr/bin/env python3
"""Exercise the actual pinned compiler. These are extraction tests, not codegen tests."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "toolchain-manifest.json").read_text())
CHANNEL = MANIFEST["rust"]["channel"]


def run(command, env, log, timeout=180):
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout,
    )
    log.write_text(result.stdout)
    return result


def require(condition, message, log=None):
    if not condition:
        detail = f"\nSee {log}" if log else ""
        raise RuntimeError(message + detail)


def parse_diagnostic(output):
    prefix = "RIPPLE_DIAGNOSTIC: "
    records = [json.loads(line[len(prefix):]) for line in output.splitlines() if line.startswith(prefix)]
    require(len(records) == 1, "expected exactly one structured Ripple diagnostic")
    require(records[0]["schema_version"] == 1, "unsupported diagnostic schema")
    return records[0]


def main():
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    summary_path = artifacts / "rust-verification.json"
    summary_path.unlink(missing_ok=True)
    for name in ("direct-marker", "helper-marker", "join-marker", "selective-marker", "arithmetic-marker", "repeat-direct-marker", "unguarded", "scalar-store"):
        (artifacts / f"{name}.json").unlink(missing_ok=True)
        (artifacts / f"{name}.c").unlink(missing_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="allow Cargo to fetch missing dependencies")
    args = parser.parse_args()
    env = os.environ.copy()
    for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_TARGET", "RUSTC"):
        env.pop(key, None)
    env["CARGO_TARGET_DIR"] = str(ROOT / "target")
    offline = [] if args.online else ["--offline"]
    version = run(["rustc", f"+{CHANNEL}", "--version", "--verbose"], env, artifacts / "rustc-version.txt")
    require(version.returncode == 0 and MANIFEST["rust"]["commit"] in version.stdout, "wrong or unavailable pinned rustc")
    build_log = artifacts / "probe-build.log"
    result = run(["cargo", f"+{CHANNEL}", "build", "--locked", *offline], env, build_log)
    require(result.returncode == 0, "probe build failed", build_log)
    env["CARGO_TARGET_DIR"] = str(ROOT / "fixtures/kernel/target")
    env["RUSTC_WORKSPACE_WRAPPER"] = str(ROOT / "target/debug/ripple-rustc-probe")
    passed = []

    # Unique report paths are deliberate Cargo inputs (fixture build.rs). This
    # prevents a warm Cargo cache from pretending it reran extraction.
    with tempfile.TemporaryDirectory(prefix="ripple-verification-", dir=artifacts) as temporary:
        temporary = Path(temporary)

        def case(name, entry="copy", feature=None, error=None, normal=False, debug=False, target=None, pointer_bodies=True, diagnostic_code=None, span_text=None):
            case_env = env.copy()
            case_env["RUSTFLAGS"] = f"-Ctarget-cpu={MANIFEST['rust']['target_cpu']}"
            if not normal:
                case_env["RUSTFLAGS"] += " --cfg ripple_frontend"
            report = temporary / f"{name}.json"
            case_env["RIPPLE_PROBE_ENTRY"] = entry
            case_env["RIPPLE_PROBE_REPORT"] = str(report)
            command = [
                "cargo", f"+{CHANNEL}", "check", "--locked", *offline,
                "-Zbuild-std=core", "-Zbuild-std-features=compiler-builtins-mem",
                "--target", target or MANIFEST["rust"]["kernel_target"],
                "--manifest-path", "fixtures/kernel/Cargo.toml",
            ]
            if not debug:
                command.append("--release")
            if feature:
                command += ["--features", feature]
            log = artifacts / f"{name}.log"
            # Seed stale content: the wrapper must remove it even if Rust rejects
            # the fixture before after_analysis. Dependency failures occur before
            # the wrapper, so the harness invalidates those reports itself.
            if error and not normal:
                report.write_text('{"stale":true}')
            result = run(command, case_env, log)
            if error:
                require(result.returncode != 0 and error in result.stdout, f"{name}: expected rejection {error!r}", log)
                require(not report.exists(), f"{name}: failure left a stale report", log)
                if diagnostic_code:
                    diagnostic = parse_diagnostic(result.stdout)
                    require(diagnostic["code"] == diagnostic_code, f"{name}: wrong diagnostic code", log)
                    require(diagnostic["source"] and "src/lib.rs:" in diagnostic["source"], f"{name}: missing source location", log)
                    if span_text:
                        lines = (ROOT / "fixtures/kernel/src/lib.rs").read_text().splitlines()
                        line = next(i + 1 for i, text in enumerate(lines) if text.strip() == span_text)
                        column = lines[line - 1].index(span_text) + 1
                        expected = f"src/lib.rs:{line}:{column}: {line}:{column + len(span_text)}"
                        require(diagnostic["source"] == expected, f"{name}: wrong span {diagnostic['source']!r}, expected {expected!r}", log)
                data = None
            else:
                require(result.returncode == 0 and report.exists(), f"{name}: extraction failed or was cached without a report", log)
                data = json.loads(report.read_text())
                require(data["pointer_width"] == 32 and data["ripple_codegen_validated"] is False, f"{name}: invalid target/stage report")
                require(data["target_cpu"] == MANIFEST["rust"]["target_cpu"], f"{name}: wrong target CPU")
                require(data["borrow_checked_local_bodies"] >= 3, f"{name}: missing local borrow-check evidence")
                calls = [call for function in data["functions"] for call in function["calls"]]
                require(any(c["classification"] == "lane_marker" for c in calls), f"{name}: marker disappeared")
                require(data["owned_ir"]["functions"] and data["stage"] == "typed_owned_ir", f"{name}: missing owned IR")
                if pointer_bodies:
                    require(any("*const f32" in f["argument_types"] and f["return_type"] == "*const f32" for f in data["functions"]), f"{name}: missing concrete const-pointer body")
                    require(any("*mut f32" in f["argument_types"] and f["return_type"] == "*mut f32" for f in data["functions"]), f"{name}: missing concrete mutable-pointer body")
                (artifacts / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
            passed.append(name)
            print(f"PASS {name}", flush=True)
            return data

        direct = case("direct-marker")
        helper = case("helper-marker", entry="copy_via_helper")
        case("join-marker", entry="copy_via_join")
        case("selective-marker", entry="copy_upper_half")
        case("arithmetic-marker", entry="copy_arithmetic")
        require(any(f["instance"] == "helper_lane" for f in helper["functions"]), "helper body not traversed")
        repeated = case("repeat-direct-marker")
        require(direct == repeated, "MIR inventory is not deterministic")
        passed.append("deterministic-inventory")
        for name in ("direct-marker", "helper-marker", "join-marker", "selective-marker", "arithmetic-marker"):
            log = artifacts / f"{name}-codegen.log"
            result = run([str(ROOT / "target/debug/ripple-codegen"), str(artifacts / f"{name}.json"), str(artifacts / f"{name}.c")], env, log)
            require(result.returncode == 0, f"{name}: code generation failed", log)
            passed.append(f"{name}-codegen")
        require((artifacts / "direct-marker.c").read_text() == (artifacts / "helper-marker.c").read_text(), "helper inlining changed generated transfer")
        for name, entry, expected in (("unguarded", "unguarded", "proven in-range lane"), ("scalar-store", "scalar_store", "lane index")):
            case(name, entry=entry, pointer_bodies=False)
            output = artifacts / f"{name}.c"
            output.write_text("/* stale output */")
            log = artifacts / f"{name}-codegen.log"
            result = run([str(ROOT / "target/debug/ripple-codegen"), str(artifacts / f"{name}.json"), str(output)], env, log)
            require(result.returncode != 0 and expected in result.stdout and not output.exists(), f"{name}: unsafe memory lowering was accepted or stale output retained", log)
            diagnostic = parse_diagnostic(result.stdout)
            require(diagnostic["code"] == ("RIPPLE-S001" if name == "unguarded" else "RIPPLE-S002"), f"{name}: wrong SPMD diagnostic", log)
            require(diagnostic["source"] is not None, f"{name}: missing SPMD source location", log)
            passed.append(f"{name}-codegen-rejected")
        case("borrow-error", feature="borrow-error", error="E0499")
        case("type-error", feature="type-error", error="E0308")
        case("unsupported-type", entry="wide_lane", feature="unsupported-type", error="unsupported IR type u64", diagnostic_code="RIPPLE-M001")
        case("unsupported-division", entry="unsupported_division", error="RIPPLE-M003", diagnostic_code="RIPPLE-M003", span_text="lane / 2")
        case("unsupported-shift", entry="unsupported_shift", error="unsupported binary operator Shl", diagnostic_code="RIPPLE-M002", span_text="lane << 1")
        case("unsupported-loop", entry="unsupported_loop", error="RIPPLE-I002", diagnostic_code="RIPPLE-I002")
        case("unsupported-drop-helper", entry="unsupported_drop_helper", error="RIPPLE-M004", diagnostic_code="RIPPLE-M004")
        case("missing-entry", entry="not_a_kernel", error="expected one non-generic entry")
        case("missing-dependency-body", entry="missing_body", feature="unavailable-body", error="unavailable or unsupported callee")
        case("normal-compilation-rejected", normal=True, error="normal compilation is unsupported")
        case("debug-runtime-unavailable", debug=True, error="panic_nounwind_fmt")

    summary = {
        "stage": "rust_owned_ir_and_c_generation", "passed": passed,
        "rustc_commit": MANIFEST["rust"]["commit"],
        "target": MANIFEST["rust"]["kernel_target"],
        "ripple_codegen_validated": False, "target_execution_validated": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(passed)} checks passed. Generated C is not yet target-qualified.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
        raise SystemExit(f"FAIL: {error}")
