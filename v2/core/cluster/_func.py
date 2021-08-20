from sklearn.cluster import KMeans
import numpy as np
from typing import Tuple


def k_mean_clustering(embeddings: np.ndarray, n_cluster: int) -> Tuple[dict, np.ndarray]:
    """
    clustering the embeddings
    :param embeddings:
    :param n_cluster:
    :return:
    """

    c = KMeans(n_clusters=n_cluster)
    c.fit(embeddings)
    clusters = dict()
    for idx, label in enumerate(c.labels_):
        if clusters.get(str(label)):
            clusters[str(label)].append(idx)
        else:
            clusters[str(label)] = [label]
    return clusters, c.cluster_centers_
