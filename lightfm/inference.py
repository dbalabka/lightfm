from typing import Tuple, Union

import numpy as np
from scipy import sparse as sp

from lightfm import LightFM, CYTHON_DTYPE

# Set of global variables for multiprocessing
_item_ids = np.array([])
_user_repr = np.array([])   # n_users, n_features
_user_repr_biases = np.array([])
_item_repr = np.ndarray([])  # n_features, n_items
_item_repr_biases = np.array([])


def _check_setup():
    if not (len(_user_repr)
        and len(_user_repr_biases)
        and len(_item_repr)
        and len(_item_repr_biases)):

        raise EnvironmentError('You must setup mode.batch_setup(item_ids) before using predict')


def _setup_items(item_ids):
    global _item_ids
    if item_ids is None:
        item_ids = np.array([])
    _item_ids = item_ids


def _batch_setup(model: LightFM,
                 item_features: Union[None, sp.csr_matrix]=None,
                 user_features: Union[None, sp.csr_matrix]=None):

    global _item_repr, _user_repr
    global _item_repr_biases, _user_repr_biases

    if item_features is None:
        n_items = len(model.item_biases)
        item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format='csr')

    if user_features is None:
        n_users = len(model.user_biases)
        user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format='csr')

    n_users = user_features.shape[0]
    user_features = model._construct_user_features(n_users, user_features)
    _user_repr, _user_repr_biases = _precompute_representation(
        features=user_features,
        feature_embeddings=model.user_embeddings,
        feature_biases=model.user_biases,
    )

    n_items = item_features.shape[0]
    item_features = model._construct_item_features(n_items, item_features)
    _item_repr, _item_repr_biases = _precompute_representation(
        features=item_features,
        feature_embeddings=model.item_embeddings,
        feature_biases=model.item_biases,
    )
    _item_repr = _item_repr.T


def _precompute_representation(
        features: sp.csr_matrix,
        feature_embeddings: np.ndarray,
        feature_biases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param: features           csr_matrix         [n_objects, n_features]
    :param: feature_embeddings np.ndarray(float)  [n_features, no_component]
    :param: feature_biases     np.ndarray(float)  [n_features]

    :return:
    TODO:
    tuple of
    - representation    np.ndarray(float)  [n_objects, no_component+1]
    - bias repr
    """

    representation = features.dot(feature_embeddings)
    representation_bias = features.dot(feature_biases)
    return representation, representation_bias


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

    if len(_item_ids):
        top_indices = _item_ids[top_indices]

    return top_indices, scores


def _batch_predict_for_user(user_id: int, top_k: int=50, item_ids=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: indices of items, top_k scores. All in score decreasing order.
    """
    # exclude biases from repr (last column of user_repr and last row of transposed item repr)
    user_repr = _user_repr[user_id, :]

    if item_ids is None:
        item_ids = _item_ids

    if item_ids is None or len(item_ids) == 0:
        item_repr = _item_repr
        item_repr_biases = _item_repr_biases
    else:
        item_repr = _item_repr[:, item_ids]
        item_repr_biases = _item_repr_biases[item_ids]

    scores = user_repr.dot(item_repr)
    scores += _user_repr_biases[user_id]
    scores += item_repr_biases
    return _get_top_k_scores(scores, k=top_k)


def _batch_cleanup():
    global _item_ids, _item_repr, _user_repr
    _item_ids = np.array([])
    _user_repr = np.array([])
    _item_repr = np.ndarray([])
