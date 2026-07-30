import unittest
import torch
from torchmetrics.detection import MeanAveragePrecision


class TestMetrics(unittest.TestCase):
    def test_map_metric_computation(self):
        metric = MeanAveragePrecision(class_metrics=True)
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        metric.update(preds, targets)
        res = metric.compute()

        self.assertIn("map_50", res)
        self.assertIn("mar_100", res)
        self.assertAlmostEqual(res["map_50"].item(), 1.0, places=2)
        self.assertAlmostEqual(res["mar_100"].item(), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
