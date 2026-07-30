import os
import tempfile
import unittest
from PIL import Image
from src.data.tile_images import (
    convert_yolo_to_abs,
    convert_abs_to_yolo,
    tile_image_and_labels_with_overlap,
)


class TestTiling(unittest.TestCase):
    def test_yolo_abs_conversions(self):
        cid, xmin, ymin, xmax, ymax = convert_yolo_to_abs(0, 0.5, 0.5, 0.2, 0.4, 1000, 1000)
        self.assertEqual(xmin, 400.0)
        self.assertEqual(xmax, 600.0)
        self.assertEqual(ymin, 300.0)
        self.assertEqual(ymax, 700.0)

        res = convert_abs_to_yolo(0, 100, 100, 300, 300, 500, 500)
        self.assertIsNotNone(res)
        cid_out, xc, yc, w, h = res
        self.assertEqual(cid_out, 0)
        self.assertAlmostEqual(xc, 0.4, places=2)
        self.assertAlmostEqual(yc, 0.4, places=2)
        self.assertAlmostEqual(w, 0.4, places=2)
        self.assertAlmostEqual(h, 0.4, places=2)

    def test_tiling_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            lbl_path = os.path.join(tmpdir, "test.txt")
            out_img = os.path.join(tmpdir, "out_img")
            out_lbl = os.path.join(tmpdir, "out_lbl")
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)

            Image.new("RGB", (1000, 1000), color="red").save(img_path)
            with open(lbl_path, "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

            tile_image_and_labels_with_overlap(
                img_path, lbl_path, out_img, out_lbl,
                tile_size=512, overlap=100,
                allow_partial_tiles=True, skip_empty_tiles=True,
            )

            tiles = os.listdir(out_img)
            self.assertTrue(len(tiles) > 0)


if __name__ == "__main__":
    unittest.main()
