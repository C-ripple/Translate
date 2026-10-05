#!/usr/bin/env python3
"""Build pinned native Ripple Clang/lld for Hexagon; keep all outputs in artifacts/."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3cc53350b1223fae796d460c108af3d9e3c96103"
SOURCE_SHA256 = "a028f84f7072c2904199141514be948fe73609bc2c8cfc4cfd20e962c06e1f48"
SOURCE_PARTS = {"llvm", "clang", "lld", "cmake", "third-party"}
PORTABILITY_FILE = "llvm/lib/Transforms/Ripple/Ripple.cpp"
# Match the declared foreachIndex/getOffsetAt interface on Darwin, where
# uint64_t and size_t have distinct C++ types. No target arithmetic changes.
PORTABILITY_REPLACEMENTS = {
    "[&](ArrayRef<TensorShape::DimSize> ToMultiIndex)": "[&](ArrayRef<size_t> ToMultiIndex)",
    "SmallVector<TensorShape::DimSize> FromMultiIndex(fromShape.rank(),": "SmallVector<size_t> FromMultiIndex(fromShape.rank(),",
}


def portable_source(content):
    text = content.decode()
    for before, after in PORTABILITY_REPLACEMENTS.items():
        if text.count(before) != 1:
            raise RuntimeError("pinned portability patch no longer applies")
        text = text.replace(before, after)
    return text.encode()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_source(archive, source):
    """Compare cached source bytes/kinds to the pinned archive, including extras."""
    expected = set()
    with tarfile.open(archive, "r|gz") as tar:
        for entry in tar:
            parts = Path(entry.name).parts
            if len(parts) < 2 or parts[1] not in SOURCE_PARTS:
                continue
            relative = Path(*parts[1:])
            path = source / relative
            expected.add(relative)
            if entry.isdir():
                valid = path.is_dir() and not path.is_symlink()
            elif entry.issym():
                valid = path.is_symlink() and os.readlink(path) == entry.linkname
            elif entry.isfile():
                valid = path.is_file() and not path.is_symlink()
                if valid:
                    stream = tar.extractfile(entry)
                    digest = hashlib.sha256()
                    if relative.as_posix() == PORTABILITY_FILE:
                        digest.update(portable_source(stream.read()))
                    else:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    valid = sha256(path) == digest.hexdigest()
            else:
                raise RuntimeError(f"unsupported source archive entry: {entry.name}")
            if not valid:
                raise RuntimeError(f"cached source differs from pinned archive: {relative}; preserve changes and recreate the source cache")
    actual = {path.relative_to(source) for path in source.rglob("*")}
    if actual != expected:
        raise RuntimeError("cached source contains missing or extra files; preserve changes and recreate the source cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="download pinned source and install isolated build tools if needed")
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    status_path = artifacts / "ripple-host-build.json"
    status_path.unlink(missing_ok=True)
    archive = artifacts / "ripple-source.tar.gz"
    source = artifacts / f"ripple-{COMMIT}"
    build = artifacts / "ripple-build"
    tools = artifacts / "build-tools/bin"
    if args.bootstrap:
        if not (tools / "cmake").exists() or not (tools / "ninja").exists():
            subprocess.run([sys.executable, "-m", "venv", str(tools.parent)], check=True)
            subprocess.run([str(tools / "python"), "-m", "pip", "install", "cmake==3.31.6", "ninja==1.11.1.4"], check=True)
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            url = f"https://codeload.github.com/qualcomm/ripple/tar.gz/{COMMIT}"
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.rename(archive)
    if not archive.exists() or sha256(archive) != SOURCE_SHA256:
        raise RuntimeError("pinned source archive missing or checksum mismatch; use --bootstrap for a missing archive")
    if not (tools / "cmake").exists() or not (tools / "ninja").exists():
        raise RuntimeError("isolated build tools missing; use --bootstrap")
    if not source.exists():
        with tempfile.TemporaryDirectory(prefix="ripple-source-", dir=artifacts) as temporary:
            subprocess.run(["tar", "-xzf", str(archive), "-C", temporary,
                            *[f"ripple-{COMMIT}/{part}" for part in sorted(SOURCE_PARTS)]], check=True)
            (Path(temporary) / source.name).rename(source)
    patched = source / PORTABILITY_FILE
    content = patched.read_bytes()
    if all(before.encode() in content for before in PORTABILITY_REPLACEMENTS):
        patched.write_bytes(portable_source(content))
    verify_source(archive, source)
    configure = [
        str(tools / "cmake"), "-G", "Ninja", "-S", str(source / "llvm"), "-B", str(build),
        f"-DCMAKE_MAKE_PROGRAM={tools / 'ninja'}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS_RELEASE=-O0 -DNDEBUG", "-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG",
        "-DLLVM_ENABLE_PROJECTS=clang;lld", "-DLLVM_TARGETS_TO_BUILD=Hexagon",
        "-DLLVM_DEFAULT_TARGET_TRIPLE=hexagon-unknown-unknown-elf", "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_INCLUDE_TESTS=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF", "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DCLANG_INCLUDE_TESTS=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_PARALLEL_LINK_JOBS=1",
    ]
    # -O0 here affects build cost of the compiler executable, not the optimization
    # setting of kernels compiled by it. Kernel probes explicitly request -O2.
    with (artifacts / "ripple-configure.log").open("w") as log:
        subprocess.run(configure, stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [str(tools / "cmake"), "--build", str(build), "--target", "clang", "lld", "clang-resource-headers", "--parallel", "2"]
    with (artifacts / "ripple-build.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if shutil.disk_usage(artifacts).free < 3 * 1024 ** 3:
                    raise RuntimeError("build stopped to preserve 3 GiB free disk; resume after adding space")
                time.sleep(5)
            if child.returncode:
                raise RuntimeError("compiler build failed; inspect artifacts/ripple-build.log")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
    result = {"commit": COMMIT, "source_sha256": SOURCE_SHA256, "source_tree_verified": True, "configure": configure,
              "host_portability_patch": PORTABILITY_REPLACEMENTS, "patched_file_sha256": sha256(patched),
              "clang_sha256": sha256(build / "bin/clang"), "lld_sha256": sha256(build / "bin/lld"),
              "target_execution_validated": False}
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    print(status_path)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"BUILD FAILED: {error}")
