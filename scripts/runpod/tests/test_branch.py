"""Host-only checks for branch isolation and the missing-volume failure path."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("check_branch", ROOT / "scripts/runpod/check_branch.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Branch test")
        self.git("config", "user.email", "branch-test@example.invalid")
        self.write("reconstruction/runtime.py", "original\n")
        self.commit()
        self.git("switch", "-q", "-c", "docker-image")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.STDOUT)

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "test fixture")

    def test_deployment_additions_are_allowed(self):
        self.write("runpod-image/Dockerfile", "FROM example\n")
        self.write("scripts/runpod/init_volume.sh", "exit 0\n")
        self.commit()
        self.assertEqual(CHECK.check_branch(self.repo, "main")[1], [])

    def test_business_edits_must_land_on_main(self):
        self.write("reconstruction/runtime.py", "unreviewed change\n")
        self.commit()
        self.assertEqual(CHECK.check_branch(self.repo, "main")[1], ["reconstruction/runtime.py"])

    def test_business_rename_cannot_bypass_the_scope_check(self):
        (self.repo / "runpod-image").mkdir()
        self.git("mv", "reconstruction/runtime.py", "runpod-image/runtime.py")
        self.commit()
        self.assertEqual(CHECK.check_branch(self.repo, "main")[1], ["reconstruction/runtime.py"])

    def test_business_changes_synced_from_main_are_allowed(self):
        self.write("runpod-image/Dockerfile", "FROM example\n")
        self.commit()
        self.git("switch", "-q", "main")
        self.write("reconstruction/runtime.py", "reviewed change\n")
        self.commit()
        self.git("switch", "-q", "docker-image")
        self.git("merge", "--no-edit", "main")
        base, unexpected = CHECK.check_branch(self.repo, "main")
        self.assertEqual(base, self.git("rev-parse", "main").decode().strip())
        self.assertEqual(unexpected, [])


class StorageTests(unittest.TestCase):
    def test_scripts_fail_before_writing_or_downloading_without_a_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            mountpoint = bin_dir / "mountpoint"
            mountpoint.write_text("#!/bin/sh\nexit 1\n")
            mountpoint.chmod(0o755)
            env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
            for path in ("runpod-image/start.sh", "scripts/runpod/bootstrap.sh", "scripts/runpod/init_volume.sh"):
                with self.subTest(path=path):
                    result = subprocess.run(["bash", str(ROOT / path)], env=env,
                                            capture_output=True, text=True, timeout=5)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("not mounted at /vol", result.stderr)


if __name__ == "__main__":
    unittest.main()
