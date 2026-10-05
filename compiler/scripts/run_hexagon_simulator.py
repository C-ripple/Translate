#!/usr/bin/env python3
"""Run the actual SDK simulator in its pinned Linux image on Apple Silicon."""

import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 4 or sys.argv[1:3] != ["--simulated_returnval", "--"]:
        raise SystemExit("expected --simulated_returnval -- /absolute/path/to/kernel.elf")
    artifacts = (ROOT / "artifacts").resolve()
    binary = Path(sys.argv[3]).resolve(strict=True)
    relative = binary.relative_to(artifacts)
    image = json.loads((artifacts / "simulator-image.json").read_text())["image_id"]
    if not image.startswith("sha256:") or len(image) != 71:
        raise SystemExit("simulator image must be pinned by its local image ID")
    name = f"ripple-probe-{uuid.uuid4().hex}"
    command = ["docker", "run", "--rm", "--name", name, "--platform", "linux/amd64", "--network=none",
               "--mount", f"type=bind,source={artifacts},target=/work,readonly", "-w", "/tmp",
               "-e", "LD_LIBRARY_PATH=/work/hexagon-sdk/lib:/work/hexagon-sdk/lib/iss",
               image, "/work/hexagon-sdk/bin/hexagon-sim", "--mv68", "--simulated_returnval", "--", f"/work/{relative.as_posix()}"]
    try:
        status = subprocess.run(command, timeout=45).returncode
    finally:
        subprocess.run(["docker", "rm", "--force", name], timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
