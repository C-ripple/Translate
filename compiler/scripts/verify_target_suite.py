#!/usr/bin/env python3
"""Run all real target probes using the locally pinned bootstrap toolchain."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = ("handwritten", "direct-marker", "helper-marker", "join-marker", "selective-marker", "arithmetic-marker")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    summary_path = artifacts / "target-suite.json"
    summary_path.unlink(missing_ok=True)
    for name in CASES:
        for suffix in (".json", ".compile.log", ".run.log"):
            (artifacts / f"target-{name}{suffix}").unlink(missing_ok=True)
    report = {"passed": False, "cases": [], "errors": []}
    try:
        build = json.loads((artifacts / "ripple-host-build.json").read_text())
        manifest = json.loads((ROOT / "toolchain-manifest.json").read_text())
        clang = artifacts / "ripple-build/bin/clang"
        linker = ROOT / "scripts/run_hexagon_linker.py"
        if (build["commit"] != manifest["ripple_candidate"]["commit"]
                or build["source_sha256"] != manifest["ripple_candidate"]["source_archive_sha256"]
                or not build["source_tree_verified"]):
            raise RuntimeError("missing pinned source verification")
        if digest(clang) != build["clang_sha256"]:
            raise RuntimeError("compiler changed since its verified build")
        sdk_root = artifacts / "hexagon-sdk"
        sdk = json.loads((sdk_root / "extraction-manifest.json").read_text())
        if sdk["directory_sha256"] != manifest["ripple_candidate"]["sdk_directory_sha256"]:
            raise RuntimeError("SDK extraction metadata does not match the pinned archive")
        for relative, key in (("bin/ld.qcld", "linker_sha256"), ("lib/libLW.so.3", "linker_library_sha256"), ("bin/hexagon-sim", "simulator_sha256")):
            if digest(sdk_root / relative) != manifest["ripple_candidate"][key]:
                raise RuntimeError(f"SDK executable/library differs from the qualified pin: {relative}")
        for entry in sdk["files"]:
            path = sdk_root / entry["path"]
            actual = hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else digest(path)
            if actual != entry["sha256"]:
                raise RuntimeError(f"SDK file changed: {entry['path']}")
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_rust_frontend.py")], cwd=ROOT, check=True, timeout=600)
        extra = [f"-B{sdk_root / 'target'}", f"-fuse-ld={linker}"]
        for name in CASES:
            command = [sys.executable, str(ROOT / "scripts/verify_ripple_target.py"),
                       "--clang", str(clang), "--simulator", str(ROOT / "scripts/run_hexagon_simulator.py"),
                       "--extra-args-json", json.dumps(extra)]
            if name != "handwritten":
                command += ["--generated-report", str(artifacts / f"{name}.json")]
            if name == "selective-marker":
                command += ["--reference", "upper-half"]
            if name == "arithmetic-marker":
                command += ["--reference", "arithmetic"]
            executed = subprocess.run(command, cwd=ROOT, timeout=240)
            current = artifacts / "target-verification.json"
            if current.exists():
                shutil.copyfile(current, artifacts / f"target-{name}.json")
            for source, suffix in (("target-compile.log", ".compile.log"), ("target-run.log", ".run.log")):
                if (artifacts / source).exists():
                    shutil.copyfile(artifacts / source, artifacts / f"target-{name}{suffix}")
            if executed.returncode or not current.exists() or not json.loads(current.read_text())["passed"]:
                raise RuntimeError(f"target case {name} failed")
            report["cases"].append(name)
        report["passed"] = True
        report["scope"] = "ABI v1, v68/HVX128, Rust f32 lane-indexed transfer subset; simulator execution, not physical-device qualification"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        report["errors"].append(str(error))
    finally:
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS" if report["passed"] else "FAILED", summary_path)
        for error in report["errors"]:
            print(error)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
