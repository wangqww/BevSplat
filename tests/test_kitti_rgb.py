"""CPU regression tests for the public KITTI depth-to-RGB command."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


spec = importlib.util.spec_from_file_location(
    'prepare_kitti_rgb', Path(__file__).resolve().parents[1] / 'scripts/prepare_kitti_rgb.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class KittiRgbTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'KITTI'
        self.camera = self.root / 'depth_data/2011_09_26/drive/image_02'
        (self.camera / 'data').mkdir(parents=True)
        (self.camera / 'grd_depth').mkdir()
        self.source = self.camera / 'data/0000000000.png'
        self.depth_path = self.camera / 'grd_depth/0000000000_grd_depth.pt'
        self.destination = self.camera / 'grd_no_sky/0000000000.png'
        Image.new('RGB', (31, 17), (19, 83, 211)).save(self.source)
        self.depth = torch.ones(256, 1024, dtype=torch.float32)
        self.depth[:128, :512] = 0
        self.depth[200, 700] = 0.0001
        torch.save(self.depth, self.depth_path)

    def test_rgb_grid_and_mask_pixels(self):
        before = [hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.source, self.depth_path)]
        self.assertEqual(helper.process_one(self.source, self.depth_path, self.destination), 'processed')
        with Image.open(self.destination) as image:
            self.assertEqual((image.mode, image.size), ('RGB', (1024, 256)))
            pixels = np.asarray(image)
            self.assertTrue((pixels[:128, :512] == 0).all())
            self.assertTrue((pixels[200, 700] == [19, 83, 211]).all())
            self.assertTrue((pixels[-1, -1] == [19, 83, 211]).all())
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (self.source, self.depth_path)])

    def test_bilinear_resize_and_rgb_conversion(self):
        image = Image.fromarray(np.array([[10, 250]], dtype=np.uint8))
        result = np.asarray(helper.rgb_from_depth(image, torch.ones(256, 1024)))
        self.assertTrue((result[:, 0] == 10).all())
        self.assertTrue((result[:, -1] == 250).all())
        self.assertTrue((result[:, 511] > 10).all())
        self.assertTrue((result[:, 511] < 250).all())
        self.assertTrue((result[..., 0] == result[..., 2]).all())

    def test_rejects_bad_depth_values(self):
        for value in [float('nan'), float('inf'), -0.01]:
            with self.subTest(value=value):
                depth = self.depth.clone()
                depth[0, 0] = value
                with self.assertRaises(ValueError):
                    helper.rgb_from_depth(Image.new('RGB', (2, 2)), depth)

    def test_rejects_bad_depth_contract(self):
        for depth in [torch.ones(2, 2), self.depth.double(), self.depth.unsqueeze(0)]:
            with self.subTest(shape=tuple(depth.shape), dtype=depth.dtype):
                with self.assertRaises(ValueError):
                    helper.rgb_from_depth(Image.new('RGB', (2, 2)), depth)
        with self.assertRaises(TypeError):
            helper.rgb_from_depth(Image.new('RGB', (2, 2)), {'depth': self.depth})

    def test_identical_rerun_and_refuses_different_existing(self):
        helper.process_one(self.source, self.depth_path, self.destination)
        self.assertEqual(helper.process_one(self.source, self.depth_path, self.destination), 'unchanged')
        Image.new('RGB', (1024, 256), (255, 255, 255)).save(self.destination)
        original = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            helper.process_one(self.source, self.depth_path, self.destination)
        self.assertEqual(original, self.destination.read_bytes())

    def test_preview_path_and_limit(self):
        second = self.source.with_name('0000000001.png')
        Image.new('RGB', (2, 2)).save(second)
        preview = Path(self.temp.name) / 'preview'
        output = io.StringIO()
        with redirect_stdout(output):
            code = helper.main(['--kitti-root', str(self.root), '--output-root', str(preview), '--limit', '1'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['processed'], 1)
        self.assertTrue((preview / self.destination.relative_to(self.root)).is_file())
        self.assertFalse(self.destination.exists())
        self.assertEqual(helper.collect_sources(self.root, 1), [self.source])

    def test_missing_depth_reports_failure_and_continues(self):
        Image.new('RGB', (2, 2)).save(self.source.with_name('0000000001.png'))
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = helper.main(['--kitti-root', str(self.root)])
        self.assertEqual(code, 1)
        counts = json.loads(output.getvalue())
        self.assertEqual((counts['processed'], counts['failed']), (1, 1))
        self.assertIn('0000000001', errors.getvalue())

    def test_cli_rejects_negative_limit(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            helper.parse_args(['--kitti-root', str(self.root), '--limit', '-1'])


if __name__ == '__main__':
    unittest.main()
