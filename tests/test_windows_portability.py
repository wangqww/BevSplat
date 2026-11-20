"""CPU portability checks for Windows/macOS builds and case-insensitive filesystems."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "build", "dist"}


class WindowsPortabilityTests(unittest.TestCase):
    def test_no_case_colliding_source_paths(self):
        if (ROOT / ".git").exists() and shutil.which("git"):
            out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True)
            paths = [line.strip() for line in out.splitlines() if line.strip()]
        else:
            paths = [
                p.relative_to(ROOT).as_posix()
                for p in ROOT.rglob("*")
                if p.is_file() and not (_SKIP_DIRS & set(p.relative_to(ROOT).parts))
            ]
        seen: dict[str, str] = {}
        collisions: list[tuple[str, str]] = []
        for p in paths:
            folded = p.casefold()
            if folded in seen and seen[folded] != p:
                collisions.append((seen[folded], p))
            seen[folded] = p
        self.assertEqual(collisions, [])
        self.assertTrue((ROOT / "models" / "dino_fit.py").is_file())

    def test_cuda_auxiliary_headers_guard_win32_math_constants(self):
        for rel in (
            "feature_gaussian/cuda_rasterizer/auxiliary.h",
            "pano_feature_gaussian/cuda_rasterizer/auxiliary.h",
        ):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("#ifdef _WIN32", text)
            self.assertIn("_USE_MATH_DEFINES", text)
            self.assertIn("M_PI", text)
            self.assertIn("M_1_PIf32", text)


if __name__ == "__main__":
    unittest.main()
