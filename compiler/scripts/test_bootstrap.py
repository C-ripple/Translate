"""Failure-path regression tests for bootstrap metadata and cache validation."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib

from test_evidence import load


class BootstrapTests(unittest.TestCase):
    def test_linker_confines_paths_and_cleans_after_timeout(self):
        linker = load("run_hexagon_linker")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            output_dir = artifacts / "ripple-target-test"
            output_dir.mkdir(parents=True)
            (artifacts / "simulator-image.json").write_text(json.dumps({"image_id": "sha256:" + "0" * 64}))
            output = output_dir / "test.elf"
            source = output_dir / "test.o"
            args = ["link", "-o", str(output), str(source), "-L" + str(artifacts / "hexagon-sdk/target")]
            with patch.object(linker, "ROOT", root), patch.object(sys, "argv", args), patch.object(linker.subprocess, "run", side_effect=[subprocess.TimeoutExpired("docker", 60), subprocess.CompletedProcess([], 0)]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    linker.main()
                command = run.call_args_list[0].args[0]
                self.assertIn("/work/ripple-target-test/test.elf", command)
                self.assertIn("/work/ripple-target-test/test.o", command)
                self.assertIn("-L/work/hexagon-sdk/target", command)
                self.assertIn(f"type=bind,source={artifacts.resolve()},target=/work,readonly", command)
                name = command[command.index("--name") + 1]
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "rm", "--force", name])
            with patch.object(linker, "ROOT", root), patch.object(sys, "argv", ["link", "-o", str(output), "/outside/object.o"]), patch.object(linker.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "outside artifacts"):
                    linker.main()
                run.assert_not_called()

    def test_failed_sdk_fetch_invalidates_old_manifest(self):
        sdk = load("fetch_hexagon_runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifacts/hexagon-sdk"
            destination.mkdir(parents=True)
            manifest = destination / "extraction-manifest.json"
            manifest.write_text('{"old":true}')
            with patch.object(sdk, "ROOT", root), patch.object(sdk, "fetch", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    sdk.main()
            self.assertFalse(manifest.exists())

    def test_sdk_repairs_executable_mode(self):
        sdk = load("fetch_hexagon_runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifacts/hexagon-sdk/bin"
            destination.mkdir(parents=True)
            executable = destination / "hexagon-sim"
            content = b"test executable bytes"
            executable.write_bytes(content)
            executable.chmod(0o644)
            entry = {"relative": "bin/hexagon-sim", "size": len(content), "crc": zlib.crc32(content), "mode": stat.S_IFREG | 0o755}
            with patch.object(sdk, "ROOT", root), patch.object(sdk, "fetch", return_value=b"metadata"), patch.object(sdk, "directory", return_value=[entry]):
                sdk.main()
            self.assertEqual(stat.S_IMODE(executable.stat().st_mode), 0o755)

    def test_wrong_kind_sdk_cache_is_not_accepted(self):
        sdk = load("fetch_hexagon_runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifacts/hexagon-sdk"
            destination.mkdir(parents=True)
            (destination / "link").write_bytes(b"target")
            entry = {"relative": "link", "name": "link", "size": 6, "crc": zlib.crc32(b"target"), "mode": stat.S_IFLNK | 0o777, "offset": 0, "compressed": 6}
            with patch.object(sdk, "ROOT", root), patch.object(sdk, "fetch", side_effect=[b"metadata", OSError("requires refetch")]), patch.object(sdk, "directory", return_value=[entry]):
                with self.assertRaisesRegex(OSError, "requires refetch"):
                    sdk.main()
            self.assertFalse((destination / "extraction-manifest.json").exists())

    def test_source_verification_detects_modified_and_extra_files(self):
        build = load("build_ripple_host")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "llvm").mkdir(parents=True)
            (source / "llvm/file").write_text("original")
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source / "llvm", arcname="ripple-pin/llvm")
            build.verify_source(archive, source)
            (source / "llvm/file").write_text("modified")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                build.verify_source(archive, source)
            (source / "llvm/file").write_text("original")
            (source / "llvm/extra").write_text("extra")
            with self.assertRaisesRegex(RuntimeError, "extra"):
                build.verify_source(archive, source)

    def test_simulator_timeout_cleans_named_container(self):
        simulator = load("run_hexagon_simulator")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            binary = artifacts / "test.elf"
            binary.write_bytes(b"test")
            (artifacts / "simulator-image.json").write_text(json.dumps({"image_id": "sha256:" + "0" * 64}))
            with patch.object(simulator, "ROOT", root), patch.object(sys, "argv", ["sim", "--simulated_returnval", "--", str(binary)]), patch.object(simulator.subprocess, "run", side_effect=[subprocess.TimeoutExpired("docker", 45), subprocess.CompletedProcess([], 0)]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    simulator.main()
                command = run.call_args_list[0].args[0]
                name = command[command.index("--name") + 1]
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "rm", "--force", name])


if __name__ == "__main__":
    unittest.main()
