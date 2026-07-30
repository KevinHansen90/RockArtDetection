import unittest
from PIL import Image
from src.data.apply_filters import (
    apply_bilateral,
    apply_unsharp,
    apply_laplacian,
    apply_clahe,
    apply_dstretch,
)


class TestFilters(unittest.TestCase):
    def test_filters_execution(self):
        img = Image.new("RGB", (100, 100), color="blue")

        bil = apply_bilateral(img)
        self.assertEqual(bil.size, (100, 100))

        unsh = apply_unsharp(img)
        self.assertEqual(unsh.size, (100, 100))

        lap = apply_laplacian(img)
        self.assertEqual(lap.size, (100, 100))

        clahe = apply_clahe(img)
        self.assertEqual(clahe.size, (100, 100))

        dst = apply_dstretch(img)
        self.assertEqual(dst.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
