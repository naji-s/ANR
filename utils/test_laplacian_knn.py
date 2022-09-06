from copy import copy
from scipy.sparse import csr_matrix
import tensorflow as tf
from numpy import float64, arange, random


@tf.function
def k_nn_graph(X, Y=None, include_self=None, k=5, mode='connectivity',
               sparse_output=True):
  """
  X: tansor of rank 2 having the refernece points with which the distance is measured with
  Y: tensor of rank 2 that their distance from the fixed set of reference points (X) is measured and the closest
  k neighbours from X is choosen
  k (int): number of neighbors to be chosen for each point in the set of points, which is stored in tensor Y
  """
  include_diagonal_ = bool(include_self)
  if (not include_diagonal_) and Y is None:
    # because now we have to choose k+1 neighbors and drop the closest one which is the self
    k = k + 1
    init_idx = 1
  else:
    init_idx = 0

  if Y is None:
    Y = copy(X)
  else:
    if include_self is not None:
      raise ValueError("if Y is not None, include_self cannot be passed")

  distance = tf.reduce_sum(tf.square(tf.subtract(Y, tf.expand_dims(X, 1))),
                           axis=-1)
  # nearest k points
  top_k_values, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)

  output = tf.zeros([tf.shape(X)[0], tf.shape(Y)[0]])
  # top_k_indices = tf.cast(top_k_indices, tf.int64)

  # for i in tf.range(tf.shape(Y)[0]):
  #     import IPython; IPython.embed()
  #     if sparse_output:
  #         sparse_indices = tf.concat([sparse_indices, tf.concat([[[i]] * tf.shape(top_k_indices[i][1-include_diagonal_:]),
  #                                                                tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1)], axis=1)], axis=0)
  #         sparse_knn_dist_values = tf.concat([sparse_knn_dist_values, tf.cast(top_k_values[i][1-include_diagonal_:], tf.float32)], axis=0)
  #     else:
  #         knn_dist_rows.append(tf.tensor_scatter_nd_update(output[i], tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1), tf.cast(top_k_values[i][1-include_diagonal_:], tf.float32)))
  #         knn_rows.append(tf.tensor_scatter_nd_update(output[i], tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1), tf.ones(tf.shape(top_k_indices[i][1-include_diagonal_:]))[0], tf.float32))
  if sparse_output:
    k_tiled = tf.tile(tf.reshape(tf.range(tf.shape(distance)[0]), (-1, 1)),
                      [1, k])[:, init_idx:]
    indices = tf.cast(
      tf.reshape(tf.stack([k_tiled, top_k_indices[:, init_idx:]], axis=-1),
                 (-1, 2)), tf.int64)
    knn = tf.SparseTensor(indices, tf.ones(tf.shape(indices)[0]),
                          [tf.shape(Y)[0], tf.shape(X)[0]])
    knn_dist = tf.negative(
      tf.SparseTensor(indices, tf.reshape(top_k_values[:, init_idx:], (-1)),
                      [tf.shape(Y)[0], tf.shape(X)[0]]))

  else:
    raise NotImplementedError("Fuck you for using not a sparse output")
  if mode == 'connectivity':
    return knn
  elif mode == 'distance':
    return knn_dist
  else:
    raise NotImplementedError


# def calculate_laplacina
def convert_sparse_tensor_to_csr_matrix(s):
  return csr_matrix(
    (s.values.numpy(), (s.indices.numpy()[:, 0], s.indices.numpy()[:, 1])),
    shape=s.get_shape().as_list(), dtype=float64)


if __name__ == '__main__':
  import numpy as np

  X = random.randn(25).reshape((5, 5))
  feature_number = tf.shape(X)
  import tensorflow as tf

  k = 2
  x_data_train = arange(30).reshape((6, 5)).astype(np.float64)
  x_data_test = arange(20).reshape((4, 5)).astype(np.float64)
  knn = k_nn_graph(X=x_data_train, k=2, sparse_output=True, mode='distance',
                   include_self=False)
  print(tf.sparse.maximum(tf.sparse.transpose(knn), knn))
  # print(tf.sparse.to_dense(tf.sparse.reorder(knn)))
