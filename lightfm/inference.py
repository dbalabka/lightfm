from typing import Tuple, Union

import numpy as np
from scipy import sparse as sp

from lightfm import LightFM

# Set of global variables for multiprocessing
_item_ids = np.array([])
_user_repr = np.array([])   # n_users, n_features
_item_repr = np.ndarray([])  # n_features, n_items


def _batch_setup(model: LightFM,
                 item_ids: np.ndarray,
                 item_features: Union[None, sp.csr_matrix]=None,
                 user_features: Union[None, sp.csr_matrix]=None):

    global _item_ids, _item_repr, _user_repr

    if item_ids.dtype != np.int32:
        item_ids = item_ids.astype(np.int32)

    n_users = user_features.shape[0]
    user_features = model._construct_user_features(n_users, user_features)
    global _user_repr
    _user_repr = _precompute_representation(
        features=user_features,
        feature_embeddings=model.user_embeddings,
        feature_biases=model.user_biases,
        scale=1.0,
        # TODO: why scale always 1.0 at the beginning?
    )

    n_items = item_features.shape[0]
    item_features = model._construct_item_features(n_items, item_features)
    _item_repr = _precompute_representation(
        item_features,
        model.item_embeddings,
        model.item_biases,
        1.0,
        # TODO: why scale always 1.0 at the beginning?
    )
    _item_repr = _item_repr.T

    _item_ids = item_ids


def _get_top_k_scores(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: indices of items, top_k scores. All in score decreasing order.
    """

    if k:
        top_indices = np.argpartition(scores, -k)[-k:]
        scores = scores[top_indices]
        sorted_top_indices = np.argsort(-scores)
        scores = scores[sorted_top_indices]
        top_indices = top_indices[sorted_top_indices]
    else:
        top_indices = np.arange(len(scores))

    return top_indices, scores


def _batch_predict_for_user(user_id: int, top_k: int=50) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: indices of items, top_k scores. All in score decreasing order.
    """
    scores = _user_repr[user_id, :].dot(_item_repr)
    return _get_top_k_scores(scores, k=top_k)


def _precompute_representation(
        features: sp.csr_matrix,
        feature_embeddings: np.ndarray,
        feature_biases: np.ndarray,
        scale: float) -> np.ndarray:
    """
    :param: features           csr_matrix         [n_objects, n_features]
    :param: feature_embeddings np.ndarray(float)  [n_features, no_component]
    :param: feature_biases     np.ndarray(float)  [n_features]
    :param: scale (learnt from factorization)

    :return: representation    np.ndarray(float)  [n_objects, no_component+1]
    """

    feature_embeddings = feature_embeddings * scale
    representation = features.dot(feature_embeddings)
    representation = np.hstack([representation, features.dot(feature_biases).reshape(-1, 1)])
    return representation


def _batch_cleanup():
    global _item_ids, _item_repr, _user_repr
    _item_ids = np.array([])
    _user_repr = np.array([])
    _item_repr = np.ndarray([])
