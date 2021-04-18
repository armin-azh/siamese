from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(embedding_1, embedding_2):
    """
    this function calculate cosine similarity between to vector
    :param embedding_1:
    :param embedding_2:
    :return:
    """
    dot = tf.reduce_sum(tf.multiply(embedding_1, embedding_2), axis=1)
    norm = tf.norm(embedding_1, axis=1) * tf.norm(embedding_2, axis=1)
    dist = tf.math.acos(dot / norm) / np.pi
    return dist


def cosine_similarity_1_k(embedding, embeddings):
    """
    this function calculate embedding cosine distance respect to embeddings
    :param embedding: vector in size (1,m)
    :param embeddings: matrix in size (n,m)
    :return: vector in shape (n,)
    """
    dot = np.sum(np.multiply(embedding, embeddings), axis=1)
    norm = np.linalg.norm(embedding, axis=1) * np.linalg.norm(embeddings, axis=1)
    dist = np.arccos(dot / norm) / np.pi
    return dist


def bulk_cosine_similarity(embeddings_1, embeddings_2):
    """
    this function calculate each vector in embeddings_1 respect to embedding_2
    :param embeddings_1:
    :param embeddings_2:
    :return:
    """
    dists = list()
    for idx in range(embeddings_1.shape[0]):
        vec = np.expand_dims(embeddings_1[idx], axis=0)
        dists.append(cosine_similarity_1_k(vec, embeddings_2))
    return np.array(dists)


def euclidean_distance(embedding, embeddings):
    """
    this function calculate embedding distance respect to embeddings
    :param embedding: (1,128)
    :param embeddings: (m,128)
    :return: vector in shape (m,)
    """
    delta = np.subtract(embedding, embeddings)
    return np.sum(np.square(delta), axis=1)


def bulk_euclidean_distance(embeddings_1, embeddings_2):
    """
    this function calculate each vector in embeddings_1 respect to embeddings_2
    :param embeddings_1: (n,128)
    :param embeddings_2: (m,128)
    :return: vector in shape (n,m)
    """

    dists = list()
    for vec in tf.unstack(embeddings_1):
        vec = np.expand_dims(vec, axis=0)
        dists.append(euclidean_distance(vec, embeddings_2))
    return np.array(dists)


def bulk_cosine_similarity_v2(embeddings_1, embeddings_2):
    return cosine_similarity(embeddings_1, embeddings_2)


def score(probabilities, distances, db_labels):
    """

    :param probabilities: matrix in shape (n_faces,classes)
    :param distances: matrix in shape (n_faces, n_gallery_image)
    :param db_labels: (n_gallery_image)
    :return:
    """
    scores = np.zeros((probabilities.shape[0],))
    for idx in range(distances.shape[0]):
        pass
