"""Public CPU release regressions; no private controller, weights, dataset or GPU."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def strict_loader():
    tree = ast.parse((ROOT / "vigor_main/train.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_initial")
    scope = {"torch": torch, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "vigor_main/train.py", "exec"), scope)
    return scope["_load_initial"]


class CheckpointTests(unittest.TestCase):
    def test_missing_checkpoint_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            strict_loader()(nn.Linear(2, 2), "/no/such/localization-model.pth", label="eval")

    def test_complete_valid_state_loads_exactly(self):
        source = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
        target = copy.deepcopy(source)
        with torch.no_grad():
            source[0].weight.fill_(3)
        with tempfile.TemporaryDirectory() as d:
            name = Path(d) / "model.pth"
            torch.save(source.state_dict(), name)
            strict_loader()(target, str(name), label="eval")
        self.assertTrue(all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items()))

    def test_invalid_state_rejected_before_model_changes(self):
        net = nn.Linear(2, 2)
        original = copy.deepcopy(net.state_dict())
        cases = []
        missing = copy.deepcopy(original); missing.pop("bias"); cases.append(missing)
        extra = copy.deepcopy(original); extra["unknown"] = torch.ones(1); cases.append(extra)
        shape = copy.deepcopy(original); shape["bias"] = torch.ones(3); cases.append(shape)
        dtype = copy.deepcopy(original); dtype["weight"] = dtype["weight"].double(); cases.append(dtype)
        for bad in [float("nan"), float("inf")]:
            state = copy.deepcopy(original); state["weight"][0, 0] = bad; cases.append(state)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.pth"
            for state in cases:
                with self.subTest(keys=list(state)):
                    torch.save(state, path)
                    with self.assertRaises(RuntimeError):
                        strict_loader()(net, str(path), label="eval")
                    self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()))


class PortablePathsTests(unittest.TestCase):
    def test_configured_root_reaches_real_loader_api(self):
        import vigor_main.data as data
        root = Path("/configured/vigor-root")
        with patch.object(data, "VIGOR_ROOT", root), patch.object(data, "_load", return_value=("train", "val")) as loader:
            self.assertEqual(data.load_train_and_val(8), ("train", "val"))
            self.assertEqual(loader.call_args.kwargs["data_root"], root)
            data.load_test(8)
            self.assertEqual(loader.call_args.kwargs["data_root"], root)

    def test_unreadable_masked_image_reports_path(self):
        from dataLoader.Vigor_dataset_gs import VIGORDataset
        sample = object.__new__(VIGORDataset)
        sample.grd_list = ["/no/such/pano_mask_sky/missing.jpg"]
        with self.assertRaisesRegex(OSError, "pano_mask_sky/missing.jpg"):
            sample[0]

    def test_local_backbone_override_keeps_requested_model_and_options(self):
        from models.public_pretrained import load_hub_model
        with tempfile.TemporaryDirectory() as d:
            Path(d, "hubconf.py").write_text("# local fixture")
            with patch.dict(os.environ, {"BEVSPLAT_DINO_REPO": d}), patch.object(torch.hub, "load", return_value="model") as load:
                self.assertEqual(load_hub_model("facebookresearch/dino:main", "dino_resnet50", env_var="BEVSPLAT_DINO_REPO", trust_repo=True), "model")
                load.assert_called_once_with(str(Path(d).resolve()), "dino_resnet50", source="local", trust_repo=True)
        with patch.dict(os.environ, {"BEVSPLAT_DINO_REPO": "/no/such/repo"}):
            with self.assertRaises(FileNotFoundError):
                load_hub_model("unused", "dino_resnet50", env_var="BEVSPLAT_DINO_REPO")


class DepthPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("portable_depth", ROOT / "scripts/prepare_vigor_depth.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_paths_order_and_output_root_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "input"; output = Path(d) / "preview"
            images = [root / city / "pano_mask_sky" / "id,lat,lon,.jpg" for city in ["NewYork", "Chicago"]]
            for image in images:
                image.parent.mkdir(parents=True); image.write_bytes(b"fixture")
            jobs = self.module.collect_jobs(images, root, output, "cross")
            self.assertEqual(jobs[0][0], str(images[0]))
            self.assertEqual(jobs[1][1], str(output / "Chicago/UniK3D_cross_metric/id,lat,lon,_depth.npy"))
            self.assertFalse(output.exists())
            self.assertEqual(len(self.module.collect_jobs(images, root, output, "same", limit=1)), 1)
            Path(jobs[0][1]).parent.mkdir(parents=True); Path(jobs[0][1]).write_bytes(b"existing")
            with self.assertRaises(FileExistsError):self.module.collect_jobs(images, root, output, "cross")
            self.assertEqual(len(self.module.collect_jobs(images, root, output, "cross", overwrite=True)), 2)

    def test_missing_input_and_unexpected_layout_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                self.module.collect_jobs([root / "NewYork/pano_mask_sky/a.jpg"], root, root, "same")
            with self.assertRaises(ValueError):
                self.module.collect_jobs([root / "NewYork/panorama/a.jpg"], root, root, "same")

    def test_cli_help_needs_no_model_or_private_controller(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_vigor_depth.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
