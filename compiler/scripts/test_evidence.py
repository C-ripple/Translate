"""Regression tests for stale evidence; no compiler/simulator behavior is mocked as passing."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvidenceTests(unittest.TestCase):
    def test_target_suite_missing_build_invalidates_old_cases(self):
        suite = load("verify_target_suite")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "target-suite.json").write_text('{"passed":true}')
            for name in suite.CASES:
                (artifacts / f"target-{name}.json").write_text('{"passed":true}')
            with patch.object(suite, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(suite.main(), 1)
            self.assertFalse(json.loads((artifacts / "target-suite.json").read_text())["passed"])
            for name in suite.CASES:
                self.assertFalse((artifacts / f"target-{name}.json").exists())

    def test_bad_target_config_replaces_previous_success(self):
        target = load("verify_ripple_target")
        for invalid in ("{", "{}", '["ok", 42]'):
            with self.subTest(config=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts = root / "artifacts"
                artifacts.mkdir()
                report = artifacts / "target-verification.json"
                report.write_text('{"passed":true}')
                with patch.object(target, "ROOT", root), patch.object(sys, "argv", ["probe", "--extra-args-json", invalid]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(target.main(), 1)
                result = json.loads(report.read_text())
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"])

    def test_missing_tools_cannot_pass_target_gate(self):
        target = load("verify_ripple_target")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(target, "ROOT", Path(directory)), patch.object(sys, "argv", ["probe", "--clang", "/no/such/clang", "--simulator", "/no/such/simulator"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(target.main(), 1)
            report = json.loads((Path(directory) / "artifacts/target-verification.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["errors"]), 2)

    def test_failed_build_invalidates_all_previous_rust_inventories(self):
        rust = load("verify_rust_frontend")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            names = ("rust-verification", "direct-marker", "helper-marker", "repeat-direct-marker")
            for name in names:
                (artifacts / f"{name}.json").write_text('{"old":true}')
            results = [subprocess.CompletedProcess([], 0, rust.MANIFEST["rust"]["commit"]), subprocess.CompletedProcess([], 1, "deliberate build failure")]
            with patch.object(rust, "ROOT", root), patch.object(rust, "run", side_effect=results), patch.object(sys, "argv", ["probe"]):
                with self.assertRaisesRegex(RuntimeError, "probe build failed"):
                    rust.main()
            for name in names:
                self.assertFalse((artifacts / f"{name}.json").exists())


if __name__ == "__main__":
    unittest.main()
