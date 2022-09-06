import numpy as np
import tensorflow as tf
DELTA = 1e-16
from scipy.linalg import cholesky
from scipy.linalg.lapack import dtrtri

def invert_mat_with_cholesky(mat=None, with_scipy=True):
    counter = 0
    eps_diag = np.diag(np.ones(mat.shape[0], dtype=np.float64) * DELTA)
    while True:
        biased_mat = mat + (counter > 0) * eps_diag * 10 ** counter
        try:
            if with_scipy:
                mat_cholesky = cholesky(tf.cast(biased_mat, tf.double))
            else:
                mat_cholesky = tf.linalg.cholesky(biased_mat).numpy()
                if np.isnan(mat_cholesky).any():
                    print ("FUCK ME!")
                    raise ValueError("tf.linalg.cholesky is failing in invert_mat_with_cholesky")
        except Exception as e:
            counter += 1
            logging = tf.get_logger()
            logging.warning(str(e) + "inverting with cholesky is failing, trial num:" + str(counter))
            if counter > 20:
                unnorm_M_inv = np.eye(mat.shape[0]) * np.sqrt(DELTA)
                break
            continue
        break
    if counter <= 10:
        if with_scipy:
            mat_cholesky_inv = dtrtri(mat_cholesky)[0]
        else:
            mat_cholesky_inv = tf.linalg.triangular_solve(tf.cast(mat_cholesky, tf.double), tf.linalg.eye(mat_cholesky.shape[0], dtype=tf.double), lower=True).numpy().T

        unnorm_M_inv = mat_cholesky_inv @ mat_cholesky_inv.T
    return unnorm_M_inv