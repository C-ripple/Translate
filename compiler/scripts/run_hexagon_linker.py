#!/usr/bin/env python3
"""Native Clang linker adapter for the pinned SDK QCLD in Linux Docker.

Clang's temporary object files and output must be in a ripple-target-* directory
under artifacts/. The SDK is read-only; only that temporary directory is writable.
"""
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifacts = (ROOT / "artifacts").resolve()
    arguments = sys.argv[1:]
    if "-o" not in arguments or any(arg.startswith("@") for arg in arguments):
        raise SystemExit("expected explicit linker output and no response files")
    output = Path(arguments[arguments.index("-o") + 1]).resolve()
    directory = output.parent
    if directory.parent != artifacts or not directory.name.startswith("ripple-target-"):
        raise SystemExit("SDK linker output must reside in an isolated target-probe directory")
    mapped = []
    for argument in arguments:
        prefix = "-L" if argument.startswith("-L/") else ""
        path_text = argument[len(prefix):]
        if path_text.startswith("/"):
            path = Path(path_text).resolve()
            try:
                relative = path.relative_to(artifacts)
            except ValueError:
                raise SystemExit(f"linker path outside artifacts: {path}")
            mapped.append(prefix + "/work/" + relative.as_posix())
        else:
            mapped.append(argument)
    image = json.loads((artifacts / "simulator-image.json").read_text())["image_id"]
    if not image.startswith("sha256:") or len(image) != 71:
        raise SystemExit("SDK image must be pinned by its local image ID")
    name = f"ripple-link-{uuid.uuid4().hex}"
    command = ["docker", "run", "--rm", "--name", name, "--platform", "linux/amd64", "--network=none",
               "--mount", f"type=bind,source={artifacts},target=/work,readonly",
               "--mount", f"type=bind,source={directory},target=/work/{directory.name}",
               "-w", "/tmp", "-e", "LD_LIBRARY_PATH=/work/hexagon-sdk/lib", image,
               "/work/hexagon-sdk/bin/ld.qcld", *mapped]
    try:
        status = subprocess.run(command, timeout=60).returncode
    finally:
        subprocess.run(["docker", "rm", "--force", name], timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
