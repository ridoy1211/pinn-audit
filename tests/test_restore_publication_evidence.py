"""Small filesystem tests; no research data or training required."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "restore_publication_evidence.py"
if not SOURCE.exists():
    SOURCE = HERE.parent / "scripts" / "restore_publication_evidence.py"
spec = importlib.util.spec_from_file_location("restore_evidence", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, bad_index=False):
        data = b"original evidence\n"
        index = {
            "output_sha256": {
                "data.txt": "0" * 64 if bad_index else hashlib.sha256(data).hexdigest()
            }
        }
        index_bytes = json.dumps(index).encode()
        archive = self.root / "evidence.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("results/example/data.txt", data)
            z.writestr("results/example/provenance.json", index_bytes)
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "assets": [
                        {
                            "filename": archive.name,
                            "bytes": archive.stat().st_size,
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                            "files": 2,
                            "uncompressed_bytes": len(data) + len(index_bytes),
                            "result_folders": {"example": 2},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_restore_and_refuse_overwrite(self):
        manifest = self.fixture()
        target = self.root / "restored"
        receipt = module.restore(manifest, self.root, target)
        self.assertEqual(receipt["indexed_hash_entries_checked"], 1)
        self.assertEqual((target / "results/example/data.txt").read_bytes(), b"original evidence\n")
        self.assertTrue((target / "RESTORE_VERIFICATION.json").is_file())
        with self.assertRaises(FileExistsError):
            module.restore(manifest, self.root, target)

    def test_wrong_archive_hash_before_write(self):
        manifest = self.fixture()
        obj = json.loads(manifest.read_text())
        obj["assets"][0]["sha256"] = "0" * 64
        manifest.write_text(json.dumps(obj))
        target = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            module.restore(manifest, self.root, target)
        self.assertFalse(target.exists())

    def test_bad_index_cannot_issue_success_receipt(self):
        manifest = self.fixture(bad_index=True)
        target = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "Indexed artifact mismatch"):
            module.restore(manifest, self.root, target)
        self.assertFalse((target / "RESTORE_VERIFICATION.json").exists())

    def test_unsafe_paths(self):
        for name in (
            "../outside",
            "/absolute",
            "C:/outside",
            "results/a/../b",
            "results/a/CON.txt",
            "results/a\\b",
            "results/a/b.",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.safe_parts(name)


if __name__ == "__main__":
    unittest.main()
