#!/usr/bin/env python3
"""Compile/run hand-written or Rust-generated ABI probes with real tools."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def executable(value):
    if not value:
        return None
    resolved = shutil.which(value)
    return str(Path(resolved).resolve()) if resolved else None


def digest(path):
    with open(path, "rb") as handle:
        result = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = artifacts / "target-verification.json"
    # Even argument-validation failures invalidate the previous run's result.
    report_path.unlink(missing_ok=True)
    for name in ("target-compile.log", "target-run.log", "target-codegen.log"):
        (artifacts / name).unlink(missing_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default=os.getenv("RIPPLE_CLANG"))
    parser.add_argument("--simulator", default=os.getenv("HEXAGON_SIM"))
    parser.add_argument("--extra-args-json", default="[]", help="JSON array of SDK/sysroot/linker flags")
    parser.add_argument("--generated-report", type=Path, help="owned-IR report to compile through ripple-codegen before the target test")
    parser.add_argument("--reference", choices=("copy", "upper-half", "arithmetic"), default="copy", help="independent scalar reference used by the target caller")
    args = parser.parse_args()
    report = {"stage": "hand_written_ripple_abi_probe", "passed": False,
              "compiler_generated_kernel": False, "reference": args.reference, "errors": []}
    try:
        extra = json.loads(args.extra_args_json)
        if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
            raise ValueError("--extra-args-json must contain a JSON array of strings")
        clang = executable(args.clang)
        simulator = executable(args.simulator)
        if not clang:
            report["errors"].append("Ripple-enabled Clang unavailable; set RIPPLE_CLANG or --clang")
        if not simulator:
            report["errors"].append("Hexagon simulator unavailable; set HEXAGON_SIM or --simulator")
        if report["errors"]:
            return 1
        report["compiler"] = {"path": clang, "sha256": digest(clang)}
        report["simulator"] = {"path": simulator, "sha256": digest(simulator)}
        version = subprocess.run([clang, "--version"], capture_output=True, text=True, timeout=10)
        report["compiler"]["version"] = version.stdout.strip()
        if version.returncode:
            report["errors"].append("compiler version query failed")
            return 1
        profile = json.loads((ROOT / "toolchain-manifest.json").read_text())["ripple_candidate"]
        with tempfile.TemporaryDirectory(prefix="ripple-target-", dir=artifacts) as temporary:
            binary = str(Path(temporary) / "copy-probe.elf")
            source = ROOT / "probes/copy.c"
            if args.generated_report:
                report["stage"] = "rust_generated_ripple_abi_probe"
                report["compiler_generated_kernel"] = True
                source = Path(temporary) / "generated.c"
                generated = subprocess.run([str(ROOT / "target/debug/ripple-codegen"), str(args.generated_report), str(source)], capture_output=True, text=True, timeout=30)
                (artifacts / "target-codegen.log").write_text(generated.stdout + generated.stderr)
                if generated.returncode:
                    report["errors"].append("owned IR lowering failed; see target-codegen.log")
                    return 1
                report["input_report_sha256"] = digest(args.generated_report)
                report["codegen_sha256"] = digest(ROOT / "target/debug/ripple-codegen")
            report["source_sha256"] = digest(source)
            report["caller_sha256"] = digest(ROOT / "probes/driver.c")
            report["abi_header_sha256"] = digest(ROOT / "probes/launch.h")
            command = [clang, f"--target={profile['target']}", *profile["flags"],
                       "-std=c11", "-I", str(ROOT / "probes"), str(source), str(ROOT / "probes/driver.c"),
                       *extra, "-o", binary]
            if args.reference == "upper-half":
                command.insert(1, "-DRIPPLE_TEST_UPPER_HALF=1")
            if args.reference == "arithmetic":
                command.insert(1, "-DRIPPLE_TEST_ARITHMETIC=1")
            report["compile_command"] = command
            environment = os.environ.copy()
            environment["TMPDIR"] = temporary
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
            (artifacts / "target-compile.log").write_text(compiled.stdout + compiled.stderr)
            if compiled.returncode:
                report["errors"].append("real Ripple compilation failed; see target-compile.log")
                return 1
            report["elf_sha256"] = digest(binary)
            sdk_linker = artifacts / "hexagon-sdk/bin/ld.qcld"
            sdk_manifest = artifacts / "hexagon-sdk/extraction-manifest.json"
            linker_adapter = ROOT / "scripts/run_hexagon_linker.py"
            using_sdk_adapter = any(arg.startswith("-fuse-ld=") and Path(arg.split("=", 1)[1]).resolve() == linker_adapter.resolve() for arg in extra)
            if using_sdk_adapter and sdk_linker.exists() and sdk_manifest.exists():
                report["linker_adapter_sha256"] = digest(linker_adapter)
                report["sdk_linker_sha256"] = digest(sdk_linker)
                report["sdk_manifest_sha256"] = digest(sdk_manifest)
            for name in ("ripple-host-build", "simulator-image"):
                provenance = artifacts / f"{name}.json"
                if provenance.exists():
                    report[name] = json.loads(provenance.read_text())
            # Same simulator invocation convention as the upstream vendored test setup.
            command = [simulator, "--simulated_returnval", "--", binary]
            report["run_command"] = command
            executed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (artifacts / "target-run.log").write_text(executed.stdout + executed.stderr)
            if executed.returncode or "RIPPLE_ABI_COPY_V1_PASS" not in executed.stdout:
                report["errors"].append("target execution failed or did not emit the success sentinel; see target-run.log")
                return 1
        report["passed"] = True
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["errors"].append(str(error))
        return 1
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS" if report["passed"] else "BLOCKED/FAILED", report_path)
        for error in report["errors"]:
            print(error)


if __name__ == "__main__":
    raise SystemExit(main())
