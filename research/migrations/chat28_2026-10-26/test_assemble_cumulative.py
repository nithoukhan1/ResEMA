"""Synthetic self-test for the OFFLINE Chat28 cumulative full/portable assembler.

Uses fabricated tiny ZIPs with fresh test hashes. It does NOT assert that
historical user ZIPs are present in this public GitHub repository.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().with_name("assemble_cumulative_from_parents.py")
SPEC = importlib.util.spec_from_file_location("chat28_assemble", PATH)
assert SPEC and SPEC.loader
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def tiny_zip(path, files):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


class Chat28CumulativeArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.full = root / "history_FULL.zip"
        self.port = root / "history_PORTABLE.zip"
        self.addon = root / "chat27_repo_PORTABLE.zip"
        self.out = root / "results"
        tiny_zip(self.full, {"historical/source.txt": b"full predecessor"})
        tiny_zip(self.port, {"historical/source.txt": b"portable predecessor"})
        docs = {
            A.HANDOFF_SOURCE + name: (name + "\n").encode()
            for name in A.EXPORT_DOCUMENTS
        }
        docs["PACKAGE_MEMBER_MANIFEST.json"] = json.dumps({
            "schema": "DETECTION_CHAT28_GITHUB_SNAPSHOT_PACKAGE_V1",
            "git_head": A.GITHUB_ARCHIVE_BUILD_SHA,
            "frozen_science_release": A.SOURCE_C1_SHA,
            "upstream_historical_zip_included": False,
        }).encode()
        tiny_zip(self.addon, docs)
        self.patches = [
            patch.object(A, "HISTORICAL_FULL_SHA256", A.sha256_file(self.full)),
            patch.object(A, "HISTORICAL_PORTABLE_SHA256", A.sha256_file(self.port)),
            patch.object(A, "NEW_RESEARCH_CODE_ZIP_SHA256", A.sha256_file(self.addon)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_two_cumulative_variants_and_checksum_registry(self):
        report = A.assemble(parent_full=self.full, parent_portable=self.port,
                            addon=self.addon, output_dir=self.out)
        self.assertEqual(report["status"], "FULL_AND_PORTABLE_ZIP_CRC_AND_MEMBER_SHA256_PASS")
        self.assertEqual(set(report["outputs"]), {"FINAL", "PORTABLE"})
        for kind, parent in (("FINAL", self.full), ("PORTABLE", self.port)):
            proof = report["outputs"][kind]
            pkg = self.out / proof["filename"]
            self.assertEqual(A.sha256_file(pkg), proof["sha256"])
            self.assertEqual(proof["zip_crc"], "PASS")
            with zipfile.ZipFile(pkg) as z:
                self.assertEqual(z.read("historical_parent/" + parent.name),
                                 parent.read_bytes())
                self.assertEqual(z.read("chat27_repo_and_handoff/" + self.addon.name),
                                 self.addon.read_bytes())
                self.assertIn("CURRENT_HANDOFF/CHAT28_FIRST_PROMPT_2026-10-27.txt",
                              z.namelist())
                self.assertEqual(z.testzip(), None)
            self.assertTrue((self.out / (pkg.name + ".sha256")).is_file())
        self.assertTrue((self.out / "MIGRATION_SHA256_CHECKSUMS.txt").is_file())
        self.assertTrue((self.out / "MIGRATION_VERIFICATION_REPORT.json").is_file())

    def test_tampered_parent_rejected_before_any_output(self):
        self.full.write_bytes(self.full.read_bytes() + b"!")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            A.assemble(parent_full=self.full, parent_portable=self.port,
                       addon=self.addon, output_dir=self.out)
        self.assertFalse(self.out.exists())

    def test_modified_addon_rejected(self):
        self.addon.write_bytes(self.addon.read_bytes() + b"corruption")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            A.assemble(parent_full=self.full, parent_portable=self.port,
                       addon=self.addon, output_dir=self.out)


if __name__ == "__main__":
    unittest.main()
