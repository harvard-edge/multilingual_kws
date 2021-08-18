import os
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf
import sklearn.cluster

import sys

import embedding.input_data as input_data


def embedding_model(
    base_model_path=Path.home()
    / "tinyspeech_harvard/multilingual_embedding_wc/models/multilingual_context_73_0.8011",
    base_model_output="dense_2",
):
    tf.get_logger().setLevel(logging.ERROR)
    base_model = tf.keras.models.load_model(base_model_path)
    tf.get_logger().setLevel(logging.INFO)

    embedding = tf.keras.models.Model(
        name="TransferLearnedModel",
        inputs=base_model.inputs,
        outputs=base_model.get_layer(name=base_model_output).output,
    )
    embedding.trainable = False
    return embedding


def cluster_and_sort(
    keyword_samples,
    embedding_model,
    seed=123,
    n_train=50,
    n_clusters=5,
    model_settings=input_data.standard_microspeech_model_settings(label_count=761),
):
    """
    Returns:
        evaluation clips sorted by distance, cluster centers
    """

    assert len(keyword_samples) > n_train, f"{n_train} > number of keyword samples"

    rng = np.random.RandomState(seed)
    kwdata = rng.permutation(keyword_samples)

    train_clips = kwdata[:n_train]
    eval_clips = kwdata[n_train:]

    print("clustering...")
    train_spectrograms = np.array(
        [input_data.file2spec(model_settings, fp) for fp in train_clips]
    )
    feature_vectors = embedding_model.predict(train_spectrograms)

    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=seed).fit(
        feature_vectors
    )

    print(f"featurizing {len(eval_clips)} clips...")
    eval_vectors = embedding_model.predict(
        np.array([input_data.file2spec(model_settings, fp) for fp in eval_clips])
    )

    print(f"evaluating...")
    l2_distances = np.linalg.norm(
        kmeans.cluster_centers_[np.newaxis] - eval_vectors[:, np.newaxis], axis=-1
    )
    l2_from_closest_cluster = np.min(l2_distances, axis=1)

    sorting = np.argsort(l2_from_closest_cluster)
    return dict(
        sorted_clips=eval_clips[sorting],
        cluster_centers=kmeans.cluster_centers_,
        distances=l2_from_closest_cluster[sorting],
    )

