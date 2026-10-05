#!/usr/bin/env python3
"""Prepare the small pinned simulator image; SDK files stay in a read-only mount."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "libncurses5_6.3-2ubuntu0.3_amd64.deb": ("universe/n/ncurses", "58da01991712c6856af0658dc5e9cb533c1a6854ab950055ade185e35bc9c276"),
    "libtinfo5_6.3-2ubuntu0.3_amd64.deb": ("universe/n/ncurses", "4df4288404108f1a156d014e8764a064e977e34e6d44931ab60451694c03c90d"),
    "libatomic1_14.2.0-4ubuntu2~24.04.1_amd64.deb": ("main/g/gcc-14", "fe49cbbc7be753528380c724a8eef5f1e31dffa9221f692c5069048d81c7449d"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch missing pinned compatibility packages")
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    packages = artifacts / "simulator-packages"
    packages.mkdir(parents=True, exist_ok=True)
    provenance = artifacts / "simulator-image.json"
    provenance.unlink(missing_ok=True)
    for name, (path, expected) in PACKAGES.items():
        destination = packages / name
        if not destination.exists() and args.download:
            url = f"https://security.ubuntu.com/ubuntu/pool/{path}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read(1024 * 1024)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"checksum mismatch downloading {name}")
            destination.write_bytes(content)
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"missing or changed pinned package {name}; use --download for missing files")
    subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", "cuda2ripple-simulator:6.5.0.0",
                    "-f", str(ROOT / "docker/simulator.Dockerfile"), str(packages)], check=True)
    image = subprocess.check_output(["docker", "image", "inspect", "cuda2ripple-simulator:6.5.0.0", "--format", "{{.Id}}"], text=True).strip()
    data = {"image_id": image, "base_image": "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55",
            "packages": {name: digest for name, (_, digest) in PACKAGES.items()}, "target_execution_validated": False}
    provenance.write_text(json.dumps(data, indent=2) + "\n")
    print(provenance)


if __name__ == "__main__":
    main()
