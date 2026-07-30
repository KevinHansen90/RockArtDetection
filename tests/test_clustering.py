import unittest
import numpy as np
from src.clustering.cluster_motifs import perform_clustering, preprocess_feats


class TestClustering(unittest.TestCase):
    def test_clustering_algorithms(self):
        features = np.random.randn(20, 64)
        processed = preprocess_feats(features, pca_dim=16)
        self.assertEqual(processed.shape, (20, 16))

        labels, inertia = perform_clustering(processed, algo="kmeans", k=3)
        self.assertEqual(len(labels), 20)
        self.assertIsNotNone(inertia)

        labels_agg, inertia_agg = perform_clustering(processed, algo="agglomerative", k=3)
        self.assertEqual(len(labels_agg), 20)

        labels_db, inertia_db = perform_clustering(processed, algo="dbscan", k=1, eps=0.5, min_s=2)
        self.assertEqual(len(labels_db), 20)


if __name__ == "__main__":
    unittest.main()
