"""CPU regressions for TransOptimizer level/channel guards and portable DUNE hub loading."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LegacyOptimizerAndDuneTests(unittest.TestCase):
    def test_trans_optimizer_g2sp_v1_supports_level_zero_and_validates_levels(self):
        from models.swin_transformer import TransOptimizerG2SP_V1

        opt = TransOptimizerG2SP_V1([32, 16, 4]).eval()
        shapes = {
            0: (1, 256, 64, 64),
            1: (1, 32, 128, 128),
            2: (1, 16, 256, 256),
        }
        with torch.no_grad():
            for lvl, shape in shapes.items():
                out = opt(torch.zeros(shape), torch.zeros(shape), lvl)
                self.assertEqual(tuple(out.shape), (1, 3))
            for unsupported in (3, 4, -1):
                with self.assertRaises(ValueError):
                    opt(torch.zeros(1, 256, 64, 64), torch.zeros(1, 256, 64, 64), unsupported)

    def test_trans_optimizer_channel_dispatches_reject_unknown_channels(self):
        from models.swin_transformer import TransOptimizerG2SP_V2, TransOptimizerS2GP_V1

        s2gp = TransOptimizerS2GP_V1().eval()
        g2sp2 = TransOptimizerG2SP_V2().eval()
        with torch.no_grad():
            with self.assertRaises(ValueError):
                s2gp(torch.zeros(1, 32, 32, 256), torch.zeros(1, 32, 32, 256))
            with self.assertRaises(ValueError):
                g2sp2(torch.zeros(1, 32, 64, 64), torch.zeros(1, 32, 64, 64))

    def test_dune_uses_portable_hub_cache_and_env_override(self):
        import models.dune as dune_mod

        class DummyVit(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Module()
                self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14)
                self.blocks = nn.ModuleList([nn.ModuleList([nn.Identity() for _ in range(12)])])

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "dune_repo"
            repo_dir.mkdir()
            (repo_dir / "hubconf.py").write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {"BEVSPLAT_DUNE_REPO": str(repo_dir)}), mock.patch(
                "torch.hub.load", return_value=DummyVit()
            ) as hub_mock:
                model = dune_mod.DINO(model_name="vitb14")
                self.assertEqual(model.patch_size, 14)
                hub_mock.assert_called_once_with(
                    str(repo_dir.resolve()),
                    "dune_vitbase_14_448_paper_encoder",
                    source="local",
                )


if __name__ == "__main__":
    unittest.main()
