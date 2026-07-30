import unittest
import torch
from src.models.detection_models import get_detection_model
from src.training.utils import compute_total_loss


class TestModels(unittest.TestCase):
    def test_fasterrcnn_training_and_eval(self):
        model = get_detection_model("fasterrcnn", num_classes=3, config={"model_type": "fasterrcnn"})
        img = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1], dtype=torch.int64)
        }
        # Training pass
        model.train()
        loss_dict = model([img], [target])
        self.assertIn("loss_classifier", loss_dict)

        # Eval pass
        model.eval()
        with torch.no_grad():
            out = model([img])
        self.assertEqual(len(out), 1)
        self.assertIn("boxes", out[0])
        self.assertIn("scores", out[0])
        self.assertIn("labels", out[0])

    def test_retinanet_training_and_eval(self):
        model = get_detection_model("retinanet", num_classes=3, config={"model_type": "retinanet"})
        img = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1], dtype=torch.int64)
        }
        # Training pass
        model.train()
        loss_dict = model([img], [target])
        self.assertIn("classification", loss_dict)

        # Eval pass
        model.eval()
        with torch.no_grad():
            out = model([img])
        self.assertEqual(len(out), 1)
        self.assertIn("boxes", out[0])
        self.assertIn("scores", out[0])
        self.assertIn("labels", out[0])

    def test_yolov5_training_and_eval(self):
        model = get_detection_model("yolov5", num_classes=3, config={"model_type": "yolov5"})
        img = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1], dtype=torch.int64)
        }
        # Training pass
        model.train()
        loss_dict = model([img], [target])
        total_loss = compute_total_loss(loss_dict)
        self.assertTrue(isinstance(total_loss, torch.Tensor))

        # Eval pass
        model.eval()
        with torch.no_grad():
            out = model([img])
        self.assertEqual(len(out), 1)
        self.assertIn("boxes", out[0])
        self.assertIn("scores", out[0])
        self.assertIn("labels", out[0])

    def test_deformable_detr_training_and_eval(self):
        model = get_detection_model("deformable_detr", num_classes=2, config={"model_type": "deformable_detr"})
        img = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0], dtype=torch.int64)
        }
        # Training pass
        model.train()
        loss_dict = model([img], [target])
        total_loss = compute_total_loss(loss_dict)
        self.assertTrue(isinstance(total_loss, torch.Tensor))

        # Eval pass
        model.eval()
        with torch.no_grad():
            out = model([img])
        self.assertEqual(len(out), 1)
        self.assertIn("boxes", out[0])
        self.assertIn("scores", out[0])
        self.assertIn("labels", out[0])


if __name__ == "__main__":
    unittest.main()
