#!/usr/bin/env python3
"""Fetch only SDK simulator/sysroot files, verifying pinned ZIP metadata and CRCs.

This avoids storing the entire 3.1 GB archive or extracting 6.5 GB of unrelated
SDK tooling. Files remain local build artifacts; this does not redistribute SDK.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
URL = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip"
ARCHIVE_BYTES = 3086093891
DIRECTORY_OFFSET = 3082920583
DIRECTORY_BYTES = 3173286
DIRECTORY_SHA256 = "4b5bfaa88112d0b6e5396cd14b350c6aea5c698e9e90ef1933d2844a27f5991c"
PREFIX = "Hexagon_SDK/6.5.0.0/tools/HEXAGON_Tools/19.0.07/Tools/"
NOTICE = "Hexagon_SDK/6.5.0.0/tools/HEXAGON_Tools/19.0.07/NOTICE.txt"


def fetch(start, end):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("server did not honor the exact bounded range request")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise RuntimeError("incomplete or oversized range response")
    return data


def directory(data):
    if hashlib.sha256(data).hexdigest() != DIRECTORY_SHA256:
        raise RuntimeError("SDK archive metadata changed; do not silently upgrade the pin")
    entries = []
    pos = 0
    while pos < len(data):
        x = struct.unpack_from("<4s6H3L5H2L", data, pos)
        if x[0] != b"PK\x01\x02":
            raise RuntimeError("invalid central directory")
        name = data[pos + 46:pos + 46 + x[10]].decode("utf-8")
        pos += 46 + x[10] + x[11] + x[12]
        relative = name[len(PREFIX):] if name.startswith(PREFIX) else ""
        if name == NOTICE:
            relative = "NOTICE.txt"
        elif not (relative in ("bin/hexagon-sim", "bin/hexagon-link", "bin/ld.qcld") or relative.startswith(("lib/iss/", "lib/libLW.so", "lib/libprotobuf", "lib/libprotoc", "lib/libedit", "lib/libclade", "lib/libc++.so", "lib/libc++abi.so", "target/hexagon/", "lib/clang/19/lib/"))):
            continue
        if name.endswith("/"):
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("unsafe SDK path")
        entries.append(dict(name=name, relative=relative, method=x[4], crc=x[7],
                            compressed=x[8], size=x[9], offset=x[16], mode=x[15] >> 16))
    return entries


def unpack(entry, data, start, destination):
    offset = entry["offset"] - start
    header = struct.unpack_from("<4s5H3L2H", data, offset)
    if header[0] != b"PK\x03\x04" or header[2] & 1:
        raise RuntimeError("invalid or encrypted local ZIP entry")
    name = data[offset + 30:offset + 30 + header[9]].decode("utf-8")
    if name != entry["name"] or header[3] != entry["method"]:
        raise RuntimeError("local ZIP header disagrees with pinned directory")
    offset += 30 + header[9] + header[10]
    compressed = data[offset:offset + entry["compressed"]]
    if entry["method"] == 0:
        content = compressed
    elif entry["method"] == 8:
        content = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError("unsupported ZIP compression method")
    if len(content) != entry["size"] or zlib.crc32(content) != entry["crc"]:
        raise RuntimeError(f"SDK content checksum failed: {name}")
    path = destination / entry["relative"]
    if not path.resolve().is_relative_to(destination.resolve()):
        raise RuntimeError("SDK output escapes destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    if stat.S_ISLNK(entry["mode"]):
        target = content.decode("utf-8")
        if not (path.parent / target).resolve().is_relative_to(destination.resolve()):
            raise RuntimeError("SDK symlink escapes destination")
        path.unlink(missing_ok=True)
        path.symlink_to(target)
    else:
        if path.is_symlink():
            path.unlink()
        path.write_bytes(content)
        path.chmod(entry["mode"] & 0o777 or 0o644)
    return {"path": entry["relative"], "sha256": hashlib.sha256(content).hexdigest()}


def main():
    destination = ROOT / "artifacts/hexagon-sdk"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "extraction-manifest.json"
    manifest_path.unlink(missing_ok=True)
    entries = directory(fetch(DIRECTORY_OFFSET, DIRECTORY_OFFSET + DIRECTORY_BYTES - 1))
    hashes = []
    pending = []
    for entry in entries:
        path = destination / entry["relative"]
        if not path.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError("cached SDK path escapes destination")
        try:
            mode = path.lstat().st_mode
            expected_symlink = stat.S_ISLNK(entry["mode"])
            if stat.S_ISLNK(mode) != expected_symlink or not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                pending.append(entry)
                continue
            content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        except FileNotFoundError:
            pending.append(entry)
            continue
        if len(content) == entry["size"] and zlib.crc32(content) == entry["crc"]:
            if not expected_symlink:
                path.chmod(entry["mode"] & 0o777 or 0o644)
            hashes.append({"path": entry["relative"], "sha256": hashlib.sha256(content).hexdigest()})
        else:
            pending.append(entry)
    # Each entry window covers the maximum local header extra-field size. Merge
    # nearby windows so thousands of small sysroot files do not need individual
    # HTTP round trips. Keep each group below approximately 32 MB.
    groups = []
    for entry in sorted(pending, key=lambda e: e["offset"]):
        start = entry["offset"]
        end = min(ARCHIVE_BYTES - 1, start + 30 + len(entry["name"].encode()) + 65535 + entry["compressed"])
        if groups and start <= groups[-1][1] + 131072 and end - groups[-1][0] < 32 * 1024 * 1024:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(entry)
        else:
            groups.append([start, end, [entry]])
    for index, (start, end, members) in enumerate(groups):
        data = fetch(start, end)
        hashes.extend(unpack(entry, data, start, destination) for entry in members)
        print(f"SDK range {index + 1}/{len(groups)}: {len(members)} verified files", flush=True)
    manifest = {"url": URL, "archive_bytes": ARCHIVE_BYTES, "directory_sha256": DIRECTORY_SHA256,
                "tool_version": "19.0.07", "files": sorted(hashes, key=lambda f: f["path"])}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
