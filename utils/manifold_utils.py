from sklearn.neighbors import kneighbors_graph
import numpy as np
from scipy.sparse import csgraph, csr_matrix
from numpy.linalg import LinAlgError, matrix_power
from time import time
from matrix_utils import invert_mat_with_cholesky
from scipy.sparse.linalg import svds, eigsh
import tensorflow as tf
import tensorflow_probability as tfp
def svd_decompose(Lap_mat, padding=False):
    
    try:
        svd_func = lambda x: eigsh(A=x, k=Lap_mat.shape[0]-1, maxiter=50000)
        D, U = svd_func(Lap_mat)
    except Exception as e:
        raise type(e)("svd_func in svd_decompose() raises this error:" + str(e))
    if padding:
        U = np.hstack((U, np.ones(D.shape[0] + 1).reshape((-1, 1)) / np.sqrt(D.shape[0]+1)))
        D = np.hstack((D, np.zeros(1))) 
    decomposition_dict = dict()
    decomposition_dict['D_vec'] = D
    decomposition_dict['U'] = U
    return decomposition_dict    

def build_W(features, k_neighbours, lengthscale, connectivity):
    
    try:
        if connectivity=='connectivity':
            W = kneighbors_graph(features, k_neighbours, mode='connectivity', include_self=False)
            W = (((W + W.T) > 0) * 1.)
            W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)

        elif connectivity=='distance':
            if type(features).__name__ != 'ndarray':
                if tf.keras.backend.shape(features)[0] >= 1:
                    features = features.numpy()
            W = kneighbors_graph(features, k_neighbours, mode='distance',include_self=False)
            W = W.maximum(W.T)
            W.data = np.square(W.data).ravel() 
            W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)
            W.data = tf.math.exp(-W.data).numpy()
            W.data = np.nan_to_num(W.data)

    except Exception as e:
        raise type(e)(str(e) + 'error is in kneighbors_graph')

    # THE SPARSE CASE USING TENSORFLOW. k_nn_graph IMPLEMENTATION CURRENTLY IS BROKEN
    # W = k_nn_graph(features, k=self.manifold_kernel_k, mode=self.opt['neighbor_mode'], include_self=False)
    # W = tf.sparse.maximum(W, tf.sparse.transpose(W))
    # W = convert_sparse_tensor_to_csr_matrix(W
    # checking for floating point error

    return W


def calculate_L_p_kernel(Lap_mat, manifold_kernel_noise, manifold_kernel_power, 
                         manifold_kernel_amplitude, use_eigsh=False, return_inverse=False):
    manifold_mat_inv = None

    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            largest_power = np.inf
            decomposition_dict = svd_decompose(Lap_mat, padding=True)
            U = decomposition_dict['U']
            U_T = U.T
            noisy_D = decomposition_dict['D_vec'] + manifold_kernel_noise
            D_vec_powered = noisy_D ** manifold_kernel_power
            # log_noisy_D = tf.math.log(noisy_D).numpy()
            # log_noisy_D[log_noisy_D > largest_power] = largest_power
            # log_noisy_D_too_small_p = log_noisy_D < -largest_power
            # L_p_mat = tf.linalg.matmul(tf.linalg.matmul(U, D_vec_powered), U_T).numpy()\
            #                                         * self._manifold_kernel_amplitude ** 2

            manifold_mat = U @ np.diag(D_vec_powered) @ U_T * manifold_kernel_amplitude ** 2
            if return_inverse:
                manifold_mat_inv = U @ np.diag(1. / D_vec_powered) @ U_T / manifold_kernel_amplitude ** 2

            # print ("WTF???", manifold_mat @ manifold_mat_inv)
        except Exception as e:
            raise type(e)("caclulating L_r raises this error when relying on eigsh:" + str(e))
    else:
        decomposition_dict = None
        noisy_L = Lap_mat + manifold_kernel_noise * np.eye(Lap_mat.shape[0])
        noisy_L_p = matrix_power(noisy_L, manifold_kernel_power)
        # noisy_L_p_cholesky = cholesky(noisy_L_p)
        # noisy_L_p_cholesky_inv = dtrtri(noisy_L_p_cholesky)[0]
        from scipy.sparse import csr_matrix 
        manifold_mat = noisy_L_p * manifold_kernel_amplitude ** 2
        if return_inverse:
            t = time()
            manifold_mat_inv = invert_mat_with_cholesky(noisy_L_p)
            print ("Inversion in calculate_L_p_kernel took", t-time(), "seconds...")
            manifold_mat_inv = manifold_mat_inv / manifold_kernel_amplitude ** 2


    return manifold_mat, decomposition_dict, manifold_mat_inv

def calculate_heat_kernel(Lap_mat, lbo_temperature, manifold_kernel_amplitude, use_eigsh, return_inverse=False):
    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            decomposition_dict = svd_decompose(Lap_mat)
            D_vec = decomposition_dict['D_vec']
            U = decomposition_dict['U']
            U_T = U.T
            D = tf.math.exp(-lbo_temperature  * D_vec)
        except Exception as e:
            raise type(e)(str(e)+'Exponentiating D failed')
        try:
            exp_Lap_mat = tf.linalg.matmul(tf.linalg.matmul(U, 
                                                            tf.linalg.diag(tf.experimental.numpy.ravel(D))), U_T) *\
                                                            manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = tf.linalg.matmul(
                                                tf.linalg.matmul(U, 
                                                    tf.linalg.diag(1. / tf.experimental.numpy.ravel(D))), U_T) /\
                                                        manifold_kernel_amplitude ** 2
            else: 
                exp_Lap_mat_inv = None


        except Exception as e:
            raise type(e)(str(e)+"exp_lap_mat calculation is failing WTF?!")
    else:
        decomposition_dict = None
        try:
            exp_Lap_mat = expm(-lbo_temperature * Lap_mat) * manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = expm(lbo_temperature * Lap_mat) / manifold_kernel_amplitude ** 2
            else:
                exp_Lap_mat_inv = None



        except Exception as e:
            raise type(e)("exponentation L using \
                                scipy.sparse.linalg.expm is raising this error:" +\
                          str(e) + " Lap_mat has type:"+str(type(Lap_mat)))            

    return exp_Lap_mat, decomposition_dict, exp_Lap_mat_inv


class MyKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self, kernel_func, gp_input_features, M_inv_plus_K_inv):
        self.kernel_func = kernel_func
        self.gp_input_features = gp_input_features
        self.M_inv_plus_K_inv = M_inv_plus_K_inv
        super().__init__(feature_ndims=1)#M_inv_plus_K_inv.shape[0])
    @tf.function
    def matrix(self, x, z):
        k_x = self.kernel_func.matrix(self.gp_input_features, x)
        k_z = self.kernel_func.matrix(self.gp_input_features, z)
        return self.kernel_func.matrix(x, z) -\
                                tf.linalg.matmul(tf.linalg.matmul(tf.transpose(k_x), self.M_inv_plus_K_inv), k_z)


def calculate_M_inv_plus_K_inv(kernel_mat=None, manifold_mat=None, manifold_mat_inv=None):
    if not manifold_mat_inv:
        counter = 0
        t = time()
        manifold_mat_inv = invert_mat_with_cholesky(manifold_mat)
        print ("Inversion in calculate_M_inv_plus_K_inv took", t-time(), "seconds...")

    M_inv_plus_K = kernel_mat + manifold_mat_inv

    t = time()
    M_inv_plus_K_inv = invert_mat_with_cholesky(M_inv_plus_K)
    print ("Inversion of M_inv_plus_K took", t-time(), "seconds...")

    return M_inv_plus_K_inv


def build_manifold_mat(W, manifold_kernel_noise, manifold_kernel_type, manifold_kernel_power, manifold_kernel_amplitude, lbo_temperature, manifold_kernel_normed=False, use_eigsh=False, return_inverse=True):
#        if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
    try:
        if hasattr(W.data, 'numpy'):
            tf.print ("we have nunmpy property")
            W.data = W.data.numpy()
        try:
            # tf.print ("YEAH FUCKER!!:",W.data.ndim, W.indices.ndim, W.indptr.ndim, output_stream=sys.stderr)
            Lap_mat = csgraph.laplacian(W, normed=manifold_kernel_normed).astype(np.float64).tocsc()
        except Exception as e:
            raise type(e)(str(e)+"you fucking CSRGRAPH!!!" + str(type(W)))



        if 'lbo' in manifold_kernel_type.lower():
            try:
                # manifold_mat, decomposition_dict, manifold_mat_inv =\
                            # self.calculate_heat_kernel(Lap_mat, use_eigsh=use_eigsh, return_inverse=return_inverse) 

                manifold_mat, decomposition_dict, manifold_mat_inv = \
                                    calculate_heat_kernel(Lap_mat, lbo_temperature=lbo_temperature, 
                                                          manifold_kernel_amplitude=manifold_kernel_amplitude, 
                                                          use_eigsh=use_eigsh, return_inverse=return_inverse)
            except Exception as e:
                raise type(e)("Calculating LBO raises this:" + str(e))

        elif 'laplacian' in manifold_kernel_type.lower():
            try:
                # manifold_mat, decomposition_dict, manifold_mat_inv = \
                #             self.calculate_L_p_kernel(Lap_mat, use_eigsh=use_eigsh, return_inverse=return_inverse)
                manifold_mat, decomposition_dict, manifold_mat_inv = \
                calculate_L_p_kernel(Lap_mat, manifold_kernel_noise=manifold_kernel_noise, 
                                     manifold_kernel_power=manifold_kernel_power, 
                                     manifold_kernel_amplitude=manifold_kernel_amplitude, 
                                     use_eigsh=use_eigsh, return_inverse=return_inverse)                    

            except Exception as e:
                raise type(e)("Calculating L^p raises this:" + str(e))

    except Exception as e:
        raise type(e)(str(e)+'FUCKKK!!!! 56' )
    return manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv



def calculate_mixed_kernel(original_kernel_func=None, sampled_gp_input_features=None, manifold_mat=None, manifold_mat_inv=None, 
                           Lap_mat=None, kernel_mat=None, decomposition_dict=None,
                            method='invert_I_plus_MK', with_expm_acting=None,
                           I_plus_MK_inverter = 'qr'):
    """ here we creat \tilde_K according to eq. (5) in https://www.ijcai.org/Proceedings/07/Papers/171.pdf
     as opposed just regularization offered in manifold regularization paper which the RKHS where f comes
     from has kernel k and not \tilde_k (which has L added to it)
     method: (string) either 'invert_M_first' or 'invert_I_plus_MK'         
     """

    # if invert_M_using_expm and 'laplacian' in self.manifold_kernel_type.lower():
    #     raise NotImplementedError("invert_M_using_expm cannot be true while the L^p is being used for SSL")

    # if with_expm_acting and 'laplacian' in self.manifold_kernel_type.lower():
    #     raise NotImplementedError("with_expm_acting cannot be true while the L^p is being used for SSL")


    try:
        M_inv_plus_K_inv = calculate_M_inv_plus_K_inv(kernel_mat=kernel_mat, 
                                                           manifold_mat=manifold_mat,
                                                           manifold_mat_inv=manifold_mat_inv)
    except Exception as e:
        if 'value' in type(e).__name__.lower():
            try:
                M_inv_plus_K_inv = calculate_M_inv_plus_K_inv(kernel_mat=kernel_mat, 
                                                               manifold_mat=manifold_mat, manifold_mat_inv=False)
            except Exception as e:
                raise type(e)("M_inv_plus_K_inv method is causing this failure\
                              even when manifold_mat_inv is explicitly set to False:" +str(e))
        else:
            raise type(e)("M_inv_plus_K_inv method is causing this failureand \
                                it is not a ValueError exception:" + str(e))                

    mixed_kernel_func = MyKernel(original_kernel_func, sampled_gp_input_features, M_inv_plus_K_inv)
    return mixed_kernel_func

def calculate_main_kernels(sampled_gp_input_features=None, use_eigsh=False, return_manifold_mat_inv=False,
                           manifold_kernel_k=None,manifold_kernel_lengthscale=None, 
                           manifold_neighbor_mode=None, manifold_kernel_noise=None, 
                           manifold_kernel_type=None, manifold_kernel_power=None, 
                           manifold_kernel_amplitude=None, lbo_temperature=None, 
                           manifold_kernel_normed=None, gp_kernel_type=None, 
                           gp_kernel_amplitude=None, gp_kernel_lengthscale=None):
#        if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
    #######################################################
    ### BEGIN: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
    #######################################################
    kernel_mat_dim = len(sampled_gp_input_features)
    W = build_W(sampled_gp_input_features, k_neighbours=manifold_kernel_k, lengthscale=manifold_kernel_lengthscale, connectivity=manifold_neighbor_mode)

    #######################################################
    ### END: CALCULATING THE LAPLACIAN CASE AND THE HEAT KERNEL
    #######################################################
    original_kernel_func = create_gp_kernel_func(gp_kernel_type=gp_kernel_type, 
                                                 gp_kernel_amplitude=gp_kernel_amplitude, gp_kernel_lengthscale=gp_kernel_lengthscale)
    kernel_mat = original_kernel_func.matrix(sampled_gp_input_features, sampled_gp_input_features).numpy()
    # manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv =\
                        # self.build_manifold_mat(W, use_eigsh=use_eigsh, return_inverse=return_manifold_mat_inv)
    manifold_mat, Lap_mat, decomposition_dict, manifold_mat_inv =\
                        build_manifold_mat(W, manifold_kernel_noise=manifold_kernel_noise, 
                                                manifold_kernel_type=manifold_kernel_type, 
                                                manifold_kernel_power=manifold_kernel_power, 
                                                manifold_kernel_amplitude=manifold_kernel_amplitude, 
                                                lbo_temperature=lbo_temperature, 
                                                manifold_kernel_normed=manifold_kernel_normed, 
                                                use_eigsh=use_eigsh, return_inverse=return_manifold_mat_inv)

    return kernel_mat, manifold_mat, manifold_mat_inv, Lap_mat, decomposition_dict, original_kernel_func

def create_gp_kernel_func(gp_kernel_type=None, gp_kernel_amplitude=None, gp_kernel_lengthscale=None):
    if gp_kernel_type == 'linear':
        tf_kernel_func = tfp.math.psd_kernels.Linear(bias_variance=0.
                                             , slope_variance=tf.cast(gp_kernel_amplitude, tf.float64), 
                                             shift=None, feature_ndims=1, validate_args=False, name='Linear')
        print ("WE GOT IN LINEAR BABY!!!")


    elif gp_kernel_type == 'rbf':
        tf_kernel_func = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=
                                tf.cast(gp_kernel_amplitude, tf.float64),length_scale=
                                tf.cast(tf.squeeze(gp_kernel_lengthscale),tf.float64), 
                                feature_ndims=1, validate_args=True, name='ExponentiatedQuadratic')

        print ("WE GOT IN RBF BABY!!!")
    elif gp_kernel_type == 'laplacian':
        raise NotImplementedError("TODO: Laplacian kernel is not implemented yet")
    else:
        raise NotImplementedError("kernel_type is not assigned")


    return tf_kernel_func