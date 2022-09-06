from copy import copy
from scipy.sparse import csr_matrix
import tensorflow as tf
from numpy import float64, arange, random
def k_nn_graph(X, Y=None, include_self=None, k=5, mode='connectivity', sparse_output=True):
    """
    X: tansor of rank 2 having the refernece points with which the distance is measured with
    Y: tensor of rank 2 that their distance from the fixed set of reference points (X) is measured and the closest
    k neighbours from X is choosen
    k (int): number of neighbors to be chosen for each point in the set of points, which is stored in tensor Y
    """
    if Y is None: 
        Y = copy(X)
    else:
        if include_self is not None:
            raise  ValueError("if Y is not None, include_self cannot be passed")
    include_diagonal_ = bool(include_self)
    if not include_diagonal_:
        k = k + 1
    distance = tf.reduce_sum(tf.square(tf.subtract(X, tf.expand_dims(Y, 1))), axis=2)
    # nearest k points
    top_k_values, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    output = tf.zeros([tf.shape(Y)[0], tf.shape(X)[0]])
    top_k_indices = tf.cast(top_k_indices, tf.int64)
    if sparse_output:
        sparse_indices = [[-1, -1]]
        sparse_knn_dist_values = []
    else:
        knn_rows = []
        knn_dist_rows = []
    for i in tf.range(tf.shape(Y)[0]):
        if sparse_output:
            sparse_indices = tf.concat([sparse_indices, tf.concat([[[i]] * tf.shape(top_k_indices[i][1-include_diagonal_:]),  tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1)], axis=1)], axis=0)
            sparse_knn_dist_values = tf.concat([sparse_knn_dist_values, tf.cast(top_k_values[i][1-include_diagonal_:], tf.float32)], axis=0)
        else:
            knn_dist_rows.append(tf.tensor_scatter_nd_update(output[i], tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1), tf.cast(top_k_values[i][1-include_diagonal_:], tf.float32)))
            knn_rows.append(tf.tensor_scatter_nd_update(output[i], tf.expand_dims(top_k_indices[i][1-include_diagonal_:], axis=1), tf.ones(tf.shape(top_k_indices[i][1-include_diagonal_:]))[0], tf.float32))
    
    if sparse_output:
        sparse_indices = sparse_indices[1:]
        knn = tf.SparseTensor(sparse_indices, tf.ones(tf.shape(sparse_indices)), [tf.shape(Y)[0], tf.shape(X)[0]])
        knn_dist = tf.SparseTensor(sparse_indices, sparse_knn_dist_values, [tf.shape(Y)[0], tf.shape(X)[0]])
        
    else:
        knn = tf.convert_to_tensor(knn_rows)
        knn_dist = tf.convert_to_tensor(knn_dist_rows)
        
    if mode == 'connectivity':
        return knn
    elif mode == 'distance':
        return knn_dist 
    else:
        raise NotImplementedError
        
def convert_sparse_tensor_to_csr_matrix(s):
    return csr_matrix((s.values.numpy(), (s.indices.numpy()[:, 0],s.indices.numpy()[:, 1])), shape=s.get_shape().as_list(), dtype=float64)

    
    
if __name__ == '__main__':
    import numpy as np
    X = random.randn(25).reshape((5, 5))
    feature_number = tf.shape(X)
    import tensorflow as tf
    k = 2
    x_data_train = arange(30).reshape((6, 5))
    x_data_test = arange(20).reshape((4, 5))
    knn = k_nn_graph(x_data_train, x_data_test, k=2, sparse_output=True, mode='distance')
    print(knn)
#     print(tf.sparse.to_dense(tf.sparse.reorder(knn)))
    