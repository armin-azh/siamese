import unittest
import numpy as np
import distance


class RecognitionTestCase(unittest.TestCase):

    def test_bulk_cosine_similarity(self):
        embedding_1 = np.random.random((50, 128))
        embedding_2 = np.random.random((60, 128))
        dists = distance.bulk_cosine_similarity(embedding_1, embedding_2)
        self.assertEqual(dists.shape[0], 50)
        self.assertEqual(dists.shape[1], 60)

    def test_bulk_euclidean_distance(self):
        embedds_1 = np.random.random((40, 512))
        embedds_2 = np.random.random((30, 512))
        dists = distance.bulk_euclidean_distance(embedds_1, embedds_2)
        self.assertEqual(dists.shape[0], 40)
        self.assertEqual(dists.shape[1], 30)


if __name__ == '__main__':
    unittest.main()
