from scipy.special import expit
from numpy import zeros, float64, exp
from sklearn.metrics.pairwise import euclidean_distances, check_pairwise_arrays, linear_kernel
from sklearn.utils.extmath import row_norms, safe_sparse_dot



def sparse_var(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    
    Taken from https://gist.github.com/sumartoyo/edba2eee645457a98fdf046e1b4297e4
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def spars_std(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    
    Taken from https://gist.github.com/sumartoyo/edba2eee645457a98fdf046e1b4297e4
    """
    return np.sqrt(vars(a, axis))




def conv_to_prob(logit):
    neg_X = logit < 0
    pos_X = ~neg_X
    output = zeros(logit.shape)
#     output[pos_X][output[pos_X] > 1e3] = 1e3
#     output[pos_X][output[pos_X] < -1e3] = -1e3
#     output[neg_X][output[neg_X] < -1e-2] = -1e-2

    output[pos_X] = expit(logit[pos_X])
    output[neg_X] = 1 - expit(-logit[neg_X])
    output[output > 1 - 1e-50] = 1 - 1e-50
    output[output < 1e-50] = 1e-50
    return output


def modified_rbf_kernel(X, Y=None, lengthscales=None, var=None, epsilon_order=6):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Read more in the :ref:`User Guide <rbf_kernel>`.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float, default None
        If None, defaults to 1.0 / n_features

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
#     X, Y = check_pairwise_arrays(X.astype(float64), Y.astype(float64))
    if lengthscales is None:
        raise NotImplementedError("You need to set lengthscales")
    if var is None:
        raise NotImplementedError("You need to set var")
    K = euclidean_distances(X, Y, squared=True)
    
    K *= -(1/lengthscales)**2 * 0.5
    K[K>eval('1e'+str(epsilon_order))] = eval('1e'+str(epsilon_order))
    K[K<eval('1e-'+str(epsilon_order))] = eval('1e-'+str(epsilon_order))  
    exp(K, K)  # exponentiate K in-place
    return K * var 



def modified_linear_kernel(X, Y=None, dense_output=True, var=None):
    """
    Compute the linear kernel between X and Y.
    Read more in the :ref:`User Guide <linear_kernel>`.
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), default=None
    dense_output : bool, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
        .. versionadded:: 0.20
    Returns
    -------
    Gram matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    if var is None:
        raise NotImplementedError("You need to set var")
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=dense_output) * var