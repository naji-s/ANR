# %%


# %% [markdown]
# # This notebook is to create SSLPU dataset in 2D for testing purposes of different methods

# %% [markdown]
# ## ***The purpose of this has been already changed. At this point I am tryinig to run CV over PsychM and try it on 2D data that has been created alredy***

# %%


# %%
# import numpy as np
# np.inf >= np.
import socket
server_name = socket.gethostname().split('.')[0]
if server_name == 'lov9':
    address_and_port = '192.168.6.99:8786'
elif server_name == 'lov8':
    address_and_port = '192.168.6.98:8786'
elif server_name == 'lov7':
    address_and_port = '192.168.6.97:8786'
elif server_name == 'lov6':
    address_and_port = '192.168.6.96:8786'
elif server_name == 'lov5':
    address_and_port = '192.168.6.95:8786'
elif server_name == 'lov4':
    address_and_port = '192.168.6.94:8786'
elif server_name == 'lov3':
    address_and_port = '192.168.6.93:8786'
elif server_name == 'lov2':
    address_and_port = '192.168.6.92:8786'
elif server_name == 'lov1':
    address_and_port = '192.168.6.91:8786'
elif server_name == 'lov10':
    address_and_port = '192.168.6.80:8786'
elif server_name == 'lov11':
    address_and_port = '192.168.6.81:8786'
elif server_name == 'lov12':
    address_and_port = '192.168.6.82:8786'
elif server_name == 'lov13':
    address_and_port = '192.168.6.83:8786'


# %%
server_name

# %%
# from LPUModels.MultiPsychM import MultiPsychM
# from LPUModels.MultiPsychM import MultiPsychM
import tensorflow as tf

# %%
from scipy.stats import norm as normal_dist
from scipy.stats import truncnorm

from matplotlib import pyplot as plt
import sys
# %matplotlib widget
# import tensorflow as tf
# from utils.LogUniformProduct import LogUniformProduct
# tf.keras.backend.set_floatx('float64')
# os.environ['MKL_NUM_THREADS']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.environ['NUMEXPR_NUM_THREADS']='1'
# os.environ['OMP_NUM_THREADS']='1'
# os.environ['PTHREAD_THREADS_MAX']='88'
import klepto
# client = LocalCluster()
from dask.distributed import Client
from utils.scorer_library import f1_0_score, elkan_c_as_score

from time import time

sys.path.append('/home/scratch/nshajari/psych_model/utils')
sys.path.append('/home/scratch/nshajari/psych_model/puLearning')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')
from utils.dataset_utils import read_swissprot,  read_neuroscience, create_synthetic_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DataFrameVectorizer import DataFrameVectorizer
from scipy.stats import beta, loguniform
import numpy as np
from numpy import random, isnan, mod
from numpy import random,  asarray, logspace, zeros
from utils.MultiViewStandardScaler import MultiViewStandardScaler
# from MultiPsychM import MultiPsychM

# from LPULabelEncoder import LPULabelEncoder
# from LPULabelDecoder import LPULabelDecoder
from utils.RepresentationPacker import RepresentationPacker
from utils.scorer_library import flexible_scorer

from os.path import isfile
fresh_data = False
dataset_type = 'animal_no_animal'
if dataset_type == 'synthetic':
    X_y_sample_type = 'circles'
    if fresh_data:
        subject='hth'
        from sklearn import datasets
        from math_utils import modified_rbf_kernel
        n_samples = 300

        if X_y_sample_type == 'circles':
            # CIRCLES AS SAMPLE
            X, y = datasets.make_circles(n_samples=n_samples, factor=.5,
                                                  noise=.05)
        #     y = 1 - y
        elif X_y_sample_type == 'moons':
            X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
        elif X_y_sample_type == 'rbf_kernel':
            X = np.random.randn(n_samples *2).reshape((-1, 2))
            phi_X = modified_rbf_kernel(X)
            alpha = random.randn(n_samples)
        # X = X * 5
        # X_scaler = StandardScaler(with_std=False)
        # X = X_scaler.fit_transform(X)

        # X, y = datasets.make_blobs(n_samples=n_samples, random_state=8)
        l_mean = 0
        y_l_mean = 0
        # while l_mean < .2 or y_l_mean <0.2:
        sig_X, sig_y, sig_l, real_params = create_synthetic_dataset(X, y, dim=2, sample_size=0, initial_lambda_range=[0.3, .5], initial_gamma_range=[0.1, .2])
        psych_X, psych_y, psych_l, real_params = sig_X, sig_y, sig_l, real_params
        true_gamma, true_lambda, w_1, b_1, true_alpha, true_beta = real_params
        y_l_mean = sig_y[sig_l==0].mean()
        l_mean = sig_l.mean()
        y = sig_y
        l = sig_l
        print (l_mean)
if not fresh_data:
    data_loc = '/home/scratch/nshajari/psych_model/full_data.pkl'
    if isfile(data_loc):
        tf.print("FUCK")
        dumper_instance = klepto.archives.file_archive(data_loc, cached=False, serialized=True)
        dumper_instance.load()
        data = dumper_instance
        for key in data.keys():
            if 'all_trials_' in key:
                data[key.split('all_trials_')[1]] = data[key]
        locals().update(data)
    sig_X = sig_X.astype(np.float64)
    psych_X = sig_X
else:
    sig_X, sig_y, sig_r, sig_l = read_neuroscience(mode='auton', subject=subject, model='vgg')
    psych_X, psych_y, psych_r, psych_l = read_neuroscience(mode='auton', subject=subject, model='vgg')
    y = sig_y
    l = sig_l



# %%


# %%
for i in all_trials_sig_X.mean(axis=0):
    print (i)

# %%
# sig_y.shape

# %%
# sig_X.shape

# %%



# %%
# for i, model in enumerate(data):#, key=lambda x: -x['file_name_idx'])):
#     row_iter = 50 - 1 - model['trial_num']
#     col_iter = 8 - active_model_dict[model['model_name']]
#     if row_iter == 2 and col_iter == 0:
#         bad_model = model

# %%
import dill as pickle
import os, sys
# from sklearn.preprocessing import LabelEncoder
if dataset_type == 'synthetic':
    sys.path.append(os.path.join(sys.path[0],'/home/scratch/nshajari/psych_model','/home/scratch/nshajari/psych_model/utils'))
    from utils.RepresentationPacker import RepresentationPacker
    data_loc = '/home/scratch/nshajari/psych_model/bad_models/bad_model_with_params_0.042088956800527960.35879566371046745NoneNone[_-1.92763682_-11.80094689]-5.153443019954475.pkl'

    # dumper_instance = klepto.archives.file_archive(data_loc, cached=False, serialized=True)
    # dumper_instance.load()
    # bad_model = dumper_instance

    with open(data_loc, 'rb') as f:
        bad_model = pickle.load(f)

# %%


# %%
if dataset_type == 'synthetic':
    l_y_cat_transformed_train = bad_model['l_y_cat_transformed_train']
    l_y_cat_transformed_test = bad_model['l_y_cat_transformed_test']
    encoder = bad_model['encoder']
    l_y_cat_decimal_train = encoder.inverse_transform(l_y_cat_transformed_train)
    l_y_cat_decimal_test = encoder.inverse_transform(l_y_cat_transformed_test)
    l_train, y_train = (l_y_cat_decimal_train / 2).astype(int), np.mod(l_y_cat_decimal_train, 2)
    l_train, y_train = (l_y_cat_decimal_train / 2).astype(int), np.mod(l_y_cat_decimal_train, 2)
    l_test, y_test = (l_y_cat_decimal_test / 2).astype(int), np.mod(l_y_cat_decimal_test, 2)
    l_test, y_test = (l_y_cat_decimal_test / 2).astype(int), np.mod(l_y_cat_decimal_test, 2)
    y = np.hstack((y_train, y_test))
    l = np.hstack((l_train, l_test))
    sig_X_train = bad_model['sig_X_train']
    sig_X_test = bad_model['sig_X_test']
    psych_X_train = bad_model['psych_X_train']
    psych_X_test = bad_model['psych_X_test']
    sig_X = np.vstack([sig_X_train, sig_X_test])
    psych_X = np.vstack([psych_X_train, psych_X_test])


# %%


# %%


# %%
# for key in data.keys():
#     for sample_idx, _ in enumerate(data['all_trials_sig_X']):
#         # print (sorted(data['all_trials_sig_X'][i][0][:, 0]))
#         if np.sum((np.asarray(sorted(data['all_trials_sig_X'][sample_idx][0][:, 0])) - np.asarray(sorted(sig_X[:, 0]))) ** 2)<1e-20:
#             # print (sample_idx)
#             break



# %%


# %%
sample = dict()
if dataset_type == 'synthetic' and not fresh_data:
    print ("FUCK")
    for key in data.keys():
        if key == 'encoder':
            encoder = data[key]
        elif key == 'hash_stamp':
            pass
        elif key == 'full_bootstrap_idx_arr':
            sample[key] = data[key][sample_idx][0]
        else:
            print (key)
            sample[key[11:]] = data[key][sample_idx][0]
            # exec(str(key)+'='+str(data[key][0]).replace("  ", ','))
    true_gamma, true_lambda, _, _, true_alpha, true_beta = sample['real_params']
    sig_X = sample['sig_X']
    y = sample['y']
    l = sample['l']
    
    from LPUModels.PsychMKeras import gamma_lambda_to_g_l_prime_transformer
    true_g_prime, true_l_prime = gamma_lambda_to_g_l_prime_transformer(true_gamma, true_lambda)
    true_g_prime, true_l_prime = true_g_prime.numpy(), true_l_prime.numpy()
    from scipy.special import expit
    X_min = np.min(sig_X[:, 0]) * 3
    X_max = np.max(sig_X[:, 0]) * 3
    Y_min = np.min(sig_X[:, 1]) * 3
    Y_max = np.max(sig_X[:, 1]) * 3
    X_grid = np.arange(X_min, X_max, 0.1)
    Y_grid = np.arange(Y_min, Y_max, 0.1)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    p_l_y_x_Z = []
    temp_input = dict()
    a = 0
    b = 0
    for x_, y_ in zip(X_grid, Y_grid):
        psych_input = np.hstack([x_.reshape((-1, 1))-a, y_.reshape((-1, 1))-b])
        p_l_y_x_Z.append((expit(psych_input @ true_alpha + true_beta)) * (1 - true_gamma - true_lambda) + true_gamma)
    p_l_y_x_Z = np.asarray(p_l_y_x_Z).reshape(X_grid.shape)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches((7.5, 7.5))
    ax[0, 0].scatter(sig_X[:, 0][y==1], sig_X[:, 1][y==1], alpha=0.8, c='b', marker='+')
    ax[0, 0].scatter(sig_X[:, 0][y==0], sig_X[:, 1][y==0], alpha=0.8, c='r', marker='_')
    ax[0, 0].set_title(r'$p(X, Y)$')
    ax[0, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3)

    ax[0, 1].scatter(sig_X[:, 0][l==1], sig_X[:, 1][l==1], alpha=0.8, c='b', marker='+')
    ax[0, 1].set_title(r'$p(X|l=1)$')
    ax[0, 1].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)

    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==1, y==1)], sig_X[:, 1][np.logical_and(l==1, y==1)], alpha=0.8, c='b', marker='+')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==1, y==0)], sig_X[:, 1][np.logical_and(l==1, y==0)], alpha=0.8, c='b', marker='+')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==0, y==1)], sig_X[:, 1][np.logical_and(l==0, y==1)], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==0, y==0)], sig_X[:, 1][np.logical_and(l==0, y==0)], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 1].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax[1, 1].set_title(r'$p(X,l)$')

    ax[1, 0].scatter(sig_X[:, 0][l==0], sig_X[:, 1][l==0], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)

    ax[1, 0].set_title(r'$p(X|l=0)$')
    ax[1, 0].set_xlabel(r'$X_1$')
    ax[1, 0].set_ylabel(r'$X_2$')
    ax[0, 1].set_xlabel(r'$X_1$')
    ax[0, 1].set_ylabel(r'$X_2$')
    ax[0, 0].set_xlabel(r'$X_1$')
    ax[0, 0].set_ylabel(r'$X_2$')
    ax[1, 1].set_xlabel(r'$X_1$')
    ax[1, 1].set_ylabel(r'$X_2$')
    # for ax_ in ax.flatten():
    #     ax_.tick_params(
    #         axis='x',          # changes apply to the x-axis
    #         which='both',      # both major and minor ticks are affected
    #         bottom=False,      # ticks along the bottom edge are off
    #         top=False, 
    #         right=False,
    #         left=False,# ticks along the top edge are off
    #         labelbottom=False) # labels along the bottom edge are off
    #     ax_.tick_params(
    #         axis='y',          # changes apply to the x-axis
    #         which='both',      # both major and minor ticks are affected
    #         bottom=False,      # ticks along the bottom edge are off
    #         top=False, 
    #         right=False,
    #         left=False,# ticks along the top edge are off
    #         labelbottom=False,
    #     labelleft=False)
    #     ax_.set_xlim(-1.5, 1.5)
    #     ax_.set_ylim(-1.5, 1.5)9

    # ax[1, 1].spines["top"].set_visible(False)
    # ax[1, 1].spines["right"].set_visible(False)
    # ax[1, 1].spines["left"].set_visible(False)
    # ax[1, 1].spines["bottom"].set_visible(False)
    # fig.savefig(X_y_sample_type+'_general_layout.png', transparent=True, facecolor='white')
    fig.tight_layout()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
# ax[1,0].set_xlabel(r'$X_1')
# X_dict = {'sig_input': X, 'psych_input': X}

# %%
a = truncnorm(a=[-10, -10], b= [10, 10], scale=.5, loc=np.zeros(1))

# %%
# l_val.mean()

# %%
# fig, ax = plt.subplots(1)
# real_params

# %%
from matplotlib import pyplot as plt
import sys, os
# %matplotlib widget
# import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
# os.environ['MKL_NUM_THREADS']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.environ['NUMEXPR_NUM_THREADS']='1'
# os.environ['OMP_NUM_THREADS']='1'
# os.environ['PTHREAD_THREADS_MAX']='88'
from pathos.multiprocessing import ProcessingPool
from utils.cluster_utils import start_cluster
from dask.distributed import Client

from time import time

sys.path.append('/home/scratch/nshajari/psych_model/utils')
sys.path.append('/home/scratch/nshajari/psych_model/puLearning')
sys.path.append('/home/scratch/nshajari/psych_model/LPUModels')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DataFrameVectorizer import DataFrameVectorizer
from sklearn import metrics
from numpy import genfromtxt
import numpy as np
from numpy import  hstack, asarray
from joblib import parallel_backend
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import linear_kernel
from dask_ml.model_selection import   RandomizedSearchCV
from MultiViewMinMaxScaler import MultiViewMinMaxScaler
# from PsychM import PsychM

from time import time, process_time
# from LPULabelEncoder import LPULabelEncoder        
# from LPULabelDecoder import LPULabelDecoder
from MultipleVectorizer import MultipleVectorizer
from RepresentationPacker import RepresentationPacker
from IdentityTransformer import IdentityTransformer
from sklearn.model_selection import train_test_split


l_mean = 0
y_l_mean = 0
# while l_mean < .2 or y_l_mean <0.001:
#     sig_X, sig_y, sig_l, real_params = create_synthetic_dataset(X, y, dim=2, sample_size=0, initial_lambda=0.05, initial_gamma=0.1)
#     psych_X, psych_y, psych_l, real_params = sig_X, sig_y, sig_l, real_params
#     y_l_mean = sig_y[sig_l==0].mean()
#     l_mean = sig_l.mean()
#     print (l_mean)
#     print (item)

# sig_X_scaler = StandardScaler(with_std=True)
# psych_X_scaler = StandardScaler(with_std=True)
# sig_X = sig_X_scaler.fit_transform(sig_X)
# psych_X = psych_X_scaler.fit_transform(psych_X)
# print (psych_y)
# print (sig_l)
# assert ((sig_y == psych_y).all)
# assert ((sig_l == psych_l).all)
# print (sig_y.values.shape, sig_X.shape, sig_l.values.shape)
# scaler = StandardScaler()
# sig_X = scaler.fit_transform(sig_X)
# y = sample['sig_y']
# l = sample['sig_l']
# k = np.random.randn(y.shape[-1]).reshape((-1, 1))

# print ("Real params are:", sample['real_params'])
encoder = LabelEncoder()
l_y_cat_transformed = encoder.fit_transform(2 * l.astype(int) + y.astype(int))
sig_X_train, sig_X_test, psych_X_train, psych_X_test, y_train, y_test, l_train, l_test, l_y_cat_transformed_train, l_y_cat_transformed_test = train_test_split(sig_X, psych_X,  y, l, l_y_cat_transformed, stratify=l_y_cat_transformed,test_size=0.8, shuffle=True)
# X_train, X_val,   y_train, y_val, l_train, l_val, l_y_cat_transformed_train, l_y_cat_transformed_val = train_test_split(X_train, y_train, l_train, l_y_cat_transformed_train, stratify=l_y_cat_transformed_train, test_size=0.1)
sig_X_val = sig_X_test
psych_X_val = psych_X_test
y_val = y_test
l_val = l_test
l_y_cat_transformed_val = l_y_cat_transformed_test
X_train = np.hstack([sig_X_train, psych_X_train])
X_val = np.hstack([sig_X_val, psych_X_val])
# psych_tst = 
# is_SPM=False, encoder=encoder)
import numpy as np
# parallel_type = 'local'
parallel_type=None
if parallel_type == 'local':
#     cluster = LocalCluster()
#     client = Client(cluster)
    client = Client(processes=False)
    
else:
    client = Client(address_and_port)#, local_dir="/zfsauton2/home/nshajari/dask/",)#, serializers=['pickle'],deserializers=['pickle'])


# %%
client

# %%
if dataset_type == 'synthetic':
    fig, ax = plt.subplots()
    # ax.contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], alpha=0.8, marker='+', c='b')
    ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], alpha=0.8, marker='_', c='r')
    # ax.set_xlim(-2, 3)
    # ax.set_ylim(-1.5, 2)
    # plt.axis('off')
    axis_off = False
    if axis_off:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False, 
            right=False,
            left=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False, 
            right=False,
            left=False,# ticks along the top edge are off
            labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    fig.savefig('moon_true.pdf', transparent=True)


# %%


# %%
psych_X_train.shape

# %%
from utils.IdentityTransformer import IdentityTransformer
# from sklearn.preprocessing import LabelEncoder
from utils.MultipleVectorizer import MultipleVectorizer
psych_vec=IdentityTransformer
psych_vec_params={}
sig_vec=IdentityTransformer
sig_vec_params={}
X_train_pack = np.hstack([sig_X_train, psych_X_train])
print(X_train_pack.shape)
# packer = RepresentationPacker(sig_X_train.shape[1], sig_X_train.shape[1])
# packer = RepresentationPacker(4096, 500)
# multiple_vec = MultipleVectorizer(representation_methods_dict
#                                   ={'psych_input':psych_vec, 'sig_input':sig_vec}, representation_params_dict={'psych_input':psych_vec_params, 'sig_input':sig_vec_params})
# X_train_new = multiple_vec.fit_transform(packer.fit_transform(X_train_pack))
# print((sig_X_train==X_train_pack[:, :2]).mean() )
# X_val_new = multiple_vec.transform({'psych_input': psych_X_val, 'sig_input': sig_X_val})

# %%
# # psych_model = PsychM(encoder=encoder, epochs=500, is_SPM=False, sig_reg_coeff= .0001, sig_reg_penalty='l2', psych_reg_coeff=.1, psych_reg_penalty='l1', metrics=[LEPUF1Score(2)])
# # from PsychM2 import PsychM2
# import tensorflow as tf
# from PsychM import PsychM
# # optf.keras.optimizers.RMSprop



# %%


# %%
# # l_y_cat_transformed = encoder.fit_transform(2 * l.values.astype(int) + y.values.astype(int))
# # sig_X_train, sig_X_test, psych_X_train, psych_X_test, y_train, y_test, l_train, l_test, l_y_cat_transformed_train, l_y_cat_transformed_test = train_test_split(sig_X, psych_X,  y, l, l_y_cat_transformed,stratify=l_y_cat_transformed,test_size=0.1, shuffle=True)
# # # X_train, X_val,   y_train, y_val, l_train, l_val, l_y_cat_transformed_train, l_y_cat_transformed_val = train_test_split(X_train, y_train, l_train, l_y_cat_transformed_train, stratify=l_y_cat_transformed_train, test_size=0.1)
# sig_X_val = sig_X_test
# psych_X_val = psych_X_test
# y_val = y_test
# l_val = l_test
# l_y_cat_transformed_val = l_y_cat_transformed_test


# %%
if dataset_type == 'synthetic':
    transformed_input = X_train_pack
    for transformer_name, transformer in results.best_estimator_.named_steps.items():
        print(transformer_name)
        if transformer_name in ['clf']:
            break
        transformed_input = transformer.transform(transformed_input)

# %%
if dataset_type == 'synthetic':
    new_y_train = y_train.ravel()
    new_y_score = bad_model['CV_estimator'].named_steps['clf'].predict_prob_y_given_x(transformed_input).ravel()
    metrics.roc_auc_score(new_y_train, new_y_score, average='micro')

# %%
if dataset_type == 'synthetic':
    metrics.roc_auc_score(new_y_train, new_y_score, average='macro')
    metrics.roc_auc_score(y_train, bad_model['CV_estimator'].named_steps['clf'].predict_prob_y_given_x(transformed_input), average='weighted')

# %%
    

# %%
# fig, ax =plt.subplots(1)
# ax.scatter(sig_X_train[:, 0][l_train==1], sig_X_train[:, 1][l_train==1], alpha=0.65, color='r')
# ax.scatter(sig_X_train[:, 0][l_train==0], sig_X_train[:, 1][l_train==0], alpha=0.05, color='b')


# %%

# history = psych_model.fit(X_train_new, l_y_cat_transformed_train, validation_data=[{'sig_input':sig_X_val, 'psych_input':X_val_new['psych_input']}, y_val, l_val, l_y_cat_transformed_val])
# # psych_model.fit(X, l)

# %%


# %%

# gamma_hist = psych_model.parameter_reporter.gamma_hist
# lambda_hist = psych_model.parameter_reporter.lambda_hist
# alpha_norm_1_list = psych_model.parameter_reporter.alpha_norm_1_hist
# alpha_norm_2_list = psych_model.parameter_reporter.alpha_norm_2_hist
# a_norm_1_list = psych_model.parameter_reporter.a_norm_1_hist
# a_norm_2_list = psych_model.parameter_reporter.a_norm_2_hist

# # acc = history_dict['accuracy']
# # val_acc = history_dict['val_accuracy']
# loss_list=history.history_['loss']
# val_loss_list=history.history_['val_loss']
# print (psych_model.alternate_param_reporter_list)
# all_alternating_gamma_list = []
# all_alternating_lambda_list = []
# all_alternating_loss_list = []
# all_alternating_val_loss_list = []
# all_alternating_val_f1score_list = []
# all_alternating_output_f1score_list = []
# all_alternating_output_lr_list = []
# all_alternating_output_alpha_norm_1_list = []
# all_alternating_output_a_norm_1_list = []
# all_alternating_output_alpha_norm_2_list = []
# all_alternating_output_a_norm_2_list = []

# # if psych_model.alternate_training:
# try:
#     for param_reporter, descending_epoch_list in zip(psych_model.alternate_param_reporter_list, psych_model.alternate_descend_epoch_list):
#     #     print (psych_model.alternate_descend_epoch_list)
#         all_alternating_gamma_list += list(np.asarray(param_reporter.gamma_hist))#[descending_epoch_list])
#         all_alternating_lambda_list += list(np.asarray(param_reporter.lambda_hist))#[descending_epoch_list])
#         all_alternating_output_alpha_norm_1_list += list(np.asarray(param_reporter.alpha_norm_1_hist))#[descending_epoch_list])
#         all_alternating_output_a_norm_1_list += list(np.asarray(param_reporter.a_norm_1_hist))#[descending_epoch_list])
#         all_alternating_output_alpha_norm_2_list += list(np.asarray(param_reporter.alpha_norm_2_hist))#[descending_epoch_list])
#         all_alternating_output_a_norm_2_list += list(np.asarray(param_reporter.a_norm_2_hist))#[descending_epoch_list])
# except:
#     all_alternating_gamma_list += list(np.asarray(param_reporter.gamma_hist))#[descending_epoch_list])
#     all_alternating_lambda_list += list(np.asarray(param_reporter.lambda_hist))#[descending_epoch_list])

# #     print (descending_epoch_list)
# try:
#     for hist, descending_epoch_list in zip(psych_model.alternate_history_list, psych_model.alternate_descend_epoch_list):
#     #     print (hist['loss'], descending_epoch_list)
#         all_alternating_loss_list += list(np.asarray(hist['loss']))#[descending_epoch_list])
#         all_alternating_val_loss_list += list(np.asarray(hist['val_loss']))#[descending_epoch_list])
#         all_alternating_val_f1score_list += list(np.asarray(hist['val_fbeta_score']))#[descending_epoch_list])
#         all_alternating_output_f1score_list += list(np.asarray(hist['fbeta_score']))#[descending_epoch_list])
#         all_alternating_output_lr_list += list(np.asarray(hist['lr']))#[descending_epoch_list])
# except:
#     all_alternating_loss_list += list(np.asarray(hist['loss']))#[descending_epoch_list])
#     all_alternating_val_loss_list += list(np.asarray(hist['val_loss']))#[descending_epoch_list])
#     all_alternating_val_f1score_list += list(np.asarray(hist['val_fbeta_score']))#[descending_epoch_list])
#     all_alternating_output_f1score_list += list(np.asarray(hist['fbeta_score']))#[descending_epoch_list])
#     all_alternating_output_lr_list += list(np.asarray(hist['lr']))#[descending_epoch_list])
    

# plt.rcParams["font.size"] = 15
# # epochs = range(1, len(first_alternate_loss_list + second_alternate_loss_list + loss_list) + 1)
# fig, ax = plt.subplots(6, 1)
# fig.set_size_inches((15,12))
# epochs = np.arange(len(all_alternating_loss_list+loss_list))
# ax[0].plot(epochs, all_alternating_loss_list+loss_list,  label='Training loss')
# ax[0].plot(epochs,all_alternating_val_loss_list+val_loss_list,  label='Validation loss')
# # ax[0].set_ylim(2, 3)
# ax[1].plot(epochs, all_alternating_val_f1score_list + history.history_['val_fbeta_score'], label='Val f1 score')
# ax[1].plot(epochs, all_alternating_output_f1score_list + history.history_['fbeta_score'], label='Train f1 sccore')
# # ax[1].plot(epochs, history.history['val_fbeta_score'], label='Val f1 score')
# # ax[1].plot(epochs, history.history['fbeta_score'], label='Train f1 sccore')
# ax[2].plot(all_alternating_gamma_list + gamma_hist,  label='Gamma')
# ax[2].plot(all_alternating_lambda_list + lambda_hist,  label='Lambda')
# ax[3].plot(all_alternating_output_lr_list + history.history_['lr'],  label='LR')
# ax[4].plot(all_alternating_output_a_norm_1_list + a_norm_1_list,  label='a_norm_1')
# ax[4].plot(all_alternating_output_a_norm_2_list + a_norm_2_list,  label='a_norm_2')
# ax[5].plot(all_alternating_output_alpha_norm_1_list + alpha_norm_1_list,  label='alpha_norm_1')
# ax[5].plot(all_alternating_output_alpha_norm_2_list + alpha_norm_2_list,  label='alpha_norm_2')

# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# ax[0].set_xlabel('Epochs')
# ax[1].set_xlabel('Epochs')
# ax[2].set_xlabel('Epochs')
# ax[3].set_xlabel('Epochs')
# ax[4].set_xlabel('Epochs')
# ax[5].set_xlabel('Epochs')

# ax[0].set_ylabel('Loss')
# ax[1].set_ylabel('F1 score')
# # ax[2].set_ylabel()
# ax[3].set_ylabel('lr')
# ax[0].legend()
# # ax[0].set_ylim([0, 1.])
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# ax[4].legend()
# ax[5].legend()
# # ax[0].set_yscale('log')
# ax[4].set_yscale('log')
# ax[5].set_yscale('log')
# plt.tight_layout()
# # ax[0].set_ylim(0, 5.)
# plt.show()

# # plt.figure(figsize=(12,9))
# # plt.plot(epochs, acc, 'bo', label='Training acc')
# # plt.plot(epochs, val_acc, 'b', label='Validation acc')
# # plt.title('Training and validation accuracy')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend(loc='lower right')
# # plt.ylim((0.5,1))
# # plt.show()


# %%
# from sklearn.metrics import f1_score
# f1_score(y_val, psych_model.predict_prob_y_given_x(X_val_new)>0.5)

# %%
# from NaiveLPU import NaiveLPU

# %%
# NaiveLPU = NaiveLPU(encoder=encoder, penalty='l2', tol=1e-8)

# %%
# NaiveLPU.fit(X_train_new, l_y_cat_transformed_train)

# %%
# y_pred = NaiveLPU.predict_prob_y_given_x(X_val_new) >=0.5

# %%
# # (y_pred == y_val).mean()
# from utils.LogUniformProduct import LogUniformProduct
# gp_params_pack = LogUniformProduct()
# lap_params_pack = LogUniformProduct(min_list=[1e-6, 5, 1e-6, 1e-6, 1e-6], max_list=[1e2, 100, 1e-1, 1e6, 1e6])
# gp_params_pack = gp_params_pack.rvs(10000)
# lap_params_pack = lap_params_pack.rvs(10000)


# %%
from utils import IdentityTransformer

# %%
if dataset_type in ['fake_reviews']:
        default_transformation = TfidfVectorizer
        psych_transformation = TfidfVectorizer
        psych_transformation_2 = CountVectorizer
        default_transformation_params = {'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 1)}
        psych_transformation_params = {'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 1), 'dtype':float64}
        psych_transformation_params_2 = {'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 2), 'dtype':float64}
elif dataset_type == 'swissprot':
    default_transformation = DataFrameVectorizer
    psych_transformation = DataFrameVectorizer
    psych_transformation_2 = DataFrameVectorizer
    default_transformation_params = {'vec_class':{'vectorizer': TfidfVectorizer}, 'vec_params':{'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 1)}, 'feature_separate':True}
    psych_transformation_params = {'vec_class':{'vectorizer': TfidfVectorizer}, 'vec_params':{'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 1), 'dtype':float64}, 'feature_separate':True}
    psych_transformation_params_2 = {'vec_class':{'vectorizer': CountVectorizer}, 'vec_params':{'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 2), 'dtype':float64}, 'feature_separate':True}
    
else:
    default_transformation = IdentityTransformer.IdentityTransformer
    psych_transformation = IdentityTransformer.IdentityTransformer
    psych_transformation_2 = IdentityTransformer.IdentityTransformer
    default_transformation_params = {}
    psych_transformation_params = {}
    psych_transformation_params_2 = {}
    

# %%
max_iter = 1000
n_iter =  100

clf__psych_reg_coeff = list(np.power(2., np.linspace(-5, 5, 21)))

clf__gp_kernel_lengthscale = np.power(2., np.linspace(-5, 0, 11))
clf__gp_kernel_amplitude = np.power(2., np.linspace(-5, 5, 21))

clf__manifold_kernel_lengthscale = np.power(2., np.linspace(-2, 3, 11))
clf__manifold_kernel_amplitude = np.power(2., np.linspace(-5, 0, 11))

clf__manifold_kernel_noise = np.power(2., np.linspace(-3, 0, 7))
clf__lbo_temperature = np.power(2., (np.linspace(-3, 3, 13)))

clf__ambient_to_intrinsic_amplitude_ratio = np.power(2., np.linspace(0, 5, 11))
clf__manifold_kernel_power =  np.arange(1, 5)

clf__C_c = clf__C_p = clf__C = 1. / clf__gp_kernel_amplitude ** 2
clf__gamma = 1 / (2 * clf__gp_kernel_amplitude ** 2)
clf__manifold_kernel_power = np.arange(1, 5)
clf__manifold_kernel_k = np.around(np.power(2, np.linspace(2, 6, 9))).astype(int)    

clf__gp_kernel_lengthscale = list(clf__gp_kernel_lengthscale.ravel())
clf__gp_kernel_amplitude = list(clf__gp_kernel_amplitude.ravel())
clf__manifold_kernel_noise = list(clf__manifold_kernel_noise.ravel())
clf__lbo_temperature = list(clf__lbo_temperature.ravel())
clf__manifold_kernel_lengthscale = list(clf__manifold_kernel_lengthscale.ravel())
clf__manifold_kernel_amplitude = list(clf__manifold_kernel_amplitude.ravel())
clf__ambient_to_intrinsic_amplitude_ratio = list(clf__ambient_to_intrinsic_amplitude_ratio.ravel())
clf__manifold_kernel_power = list(clf__manifold_kernel_power.ravel())
clf__C_c = list(clf__C_c.ravel())  
clf__C_p = list(clf__C_p.ravel())
clf__C = list(clf__C.ravel())
clf__gamma = list(clf__gamma.ravel())

# %%
param_grid =  {'clf__ssl_type': ['SSL_LBO'],
                             # 'clf__psych_reg_coeff': uniform(1e-4, 1e4),
                             # 'clf__gp_kernel_lengthscale': loguniform(1e-1, 1),
                             # 'clf__gp_kernel_amplitude': loguniform(1e-3, 1.),
                             # 'clf__manifold_kernel_lengthscale': loguniform(1e-1, 1),
                             # 'clf__manifold_kernel_noise': loguniform(1e-4, 1),
                             # 'clf__lbo_temperature': loguniform(1e-3, 10.),
                             # 'clf__manifold_kernel_amplitude': loguniform(1e-3, 1.),
                             # 'clf__ambient_to_intrinsic_amplitude_ratio': loguniform(1, 1e2),

                                         
                             'clf__max_iter': [max_iter],              
                             'clf__psych_reg_coeff': clf__psych_reg_coeff,
                             'clf__gp_kernel_lengthscale': clf__gp_kernel_lengthscale,
                             'clf__gp_kernel_amplitude': clf__gp_kernel_amplitude,
                             'clf__manifold_kernel_lengthscale': clf__manifold_kernel_lengthscale,
                             'clf__manifold_kernel_noise': clf__manifold_kernel_noise,
                             'clf__lbo_temperature': clf__lbo_temperature, 
                             'clf__manifold_kernel_amplitude': clf__manifold_kernel_amplitude,
                             'clf__ambient_to_intrinsic_amplitude_ratio': clf__ambient_to_intrinsic_amplitude_ratio,

                                         
                                         
                             'clf__manifold_kernel_power': clf__manifold_kernel_power,                                          
                            'clf__manifold_kernel_params_pack': [None],
                                         
                                         
                                         
                            # 'clf__psych_alpha': [w_2.reshape((-1, 1)) ],# + np.random.randn(*w_2.reshape((-1, 1)).shape).reshape((-1, 1)) * 0.01], #normal_dist(loc=0., scale=2),
                            #  'clf__psych_beta': [asarray(b_2)],# + np.random.randn() * 0.01)], #normal_dist(loc=0., scale=2),
                            #  'clf__g_prime': [real_g_prime],# + np.random.randn() * 0.01], #normal_dist(loc=0., scale=2),
                            #  'clf__l_prime': [real_l_prime],# + np.random.randn() * 0.01], #normal_dist(loc=0., scale=2),
              

                             'clf__psych_reg_penalty':[None],
                             'clf__sig_reg_coeff':[None],
                             'clf__sig_reg_penalty':[None],# keras.regularizers.L1], 
                             'clf__manifold_neighbor_mode': ['connectivity'],
                              'clf__freeze_psychometric_original': [False],
                             'clf__fresh_opt_initial_point': [True],
                             'clf__gp_kernel_lengthscale_trainable':[False],
                             'clf__manifold_kernel_lengthscale_trainable': [False],
                             'clf__gp_kernel_type': ['linear'],
                             'clf__dropout_rate':[0.0], #beta(a=0.7, b=5),#linspace(0, 1., 100, endpoint=False),# , #,
                             'clf__noise_sigma':[0.], #beta(a=0.8, b=5),
                             'clf__manifold_kernel_type': ['linear', 'rbf'],
                             'clf__manifold_kernel_normed':[False],
                             'clf__manifold_kernel_amplitude_trainable':[False],
                             'clf__manifold_kernel_k': clf__manifold_kernel_k, 
                             'clf__number_of_successful_attempts': [1], 
#                              'clf__sig_a': truncnorm(a=-1, b=1,  scale=0.1),
#                              'clf__sig_b': normal_dist(loc=0., scale=0.1),
#                              'clf__g_prime': normal_dist(loc=-1., scale=0.1), #normal_dist(loc=0., scale=2),
#                              'clf__l_prime': normal_dist(loc=-1., scale=0.1), #normal_dist(loc=0., scale=2),

                             'clf__warm_f_g_prime_init': [None],
                             'clf__warm_f_l_prime_init': [None],
                             'clf__lr_reduce_min_delta': [1e-3], 
                             'clf__is_fitted': [False], 
                             'clf__end_training_min_delta': [1e-4],
                             'clf__epochs':  [1000], 
#                                         'clf__psych_reg_penalty':[None], 
#                                         'clf__psych_vec': [default_transformation],
#                                         'clf__psych_vec_params': [default_transformation_params],#{'token_pattern':r'(?u)\b[A-Za-z]+\b', 'ngram_range': (1, 1), 'ngram_range':(1,4)}],
#                                         'clf__sig_vec': [default_transformation],
#                                         'clf__sig_vec_params': [default_transformation_params],
                             'clf__encoder': [encoder],   
                             'clf__is_SPM': [False],
                             'clf__batch_size': [-1],
                             'clf__verbose': [0],
                             'clf__constrained_optimization': [None],
                             'clf__alternate_training':[False],
                             'clf__calibrate':[False],
                             'clf__workers': [-1],
                             'clf__use_multiprocessing':[True],
                             'clf__with_laplace_method':[True],
                              
                             'clf__optimizer_dict':  [{'optimizer_name': 'lbfgs','learning_rate': 0.001}],
                             'clf__loss_type': ['expanded_using_softmax_mathematica'],
                             'clf__warm_cv_params':[None], #'clf__warm_cv_params': [ {'cv': 5, 'random':True, 'gammas': clf__gamma, 'Cs':clf__C_c, 'sample_size':1}], 
#                               'clf__warm_cv_params': [ {'cv': 5, 'random':True, 'Cs':logspace(-4, 4, 100), 'sample_size':50}], 
#                              'clf__manifold_neighbor_mode' : [None],
                             'packer__sig_input_dim': [all_trials_sig_X.shape[-1]],
                             'packer__psych_input_dim': [all_trials_psych_X.shape[-1]],                                                             'packer__normalize_psych_input': [False],                               
                             'packer__normalize_sig_input': [False],                                 
                             'packer__normalize_psych_input': [False],                                 
                             'standard_scaler__active': [True],
                             'min_max_scaler__active': [False],                                             
                            }
            
param_grid['vectorizer__representation_methods_dict'] = [{'psych_input': psych_transformation, 'sig_input': default_transformation}]
param_grid['vectorizer__representation_params_dict'] = [{'psych_input': psych_transformation_params, 'sig_input': default_transformation_params}]
                

# %%
# from sklearn.utils.validation import check_is_fitted
# print(check_is_fitted(encoder))



# %%
from PsychMKeras import g_l_prime_to_gamma_lambda_transformer

# %%
g_l_prime_to_gamma_lambda_transformer(1.098, 1.098)

# %%

# %%


# %%
def fit_model(X_train_new, l_y_cat_transformed_train, param_grid):
    sys.path.insert(0, '/home/scratch/nshajari/psych_model/')
    sys.path.insert(0, '/home/scratch/nshajari/psych_model/utils')
    sys.path.insert(0, '/home/scratch/nshajari/psych_model/') 
    sys.path.insert(0, '/home/nshajari/psych_model/') 
    sys.path.insert(0, '/home/nshajari/psych_model/miscellaneous') 
    sys.path.insert(0, '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/')         
    sys.path.insert(0, '/home/scratch/nshajari/psych_model/utils/') 
#     print ("KEYS:", X_train_new.keys())
    from TransformToCOO import TransformToCOO
    from MultiViewStandardScaler import MultiViewStandardScaler
    from MultiViewMinMaxScaler import MultiViewMinMaxScaler
    from PsychM import PsychM
    
    #     from MultiPythonPsychM import MultiPythonPsychM

    from LogUniformProduct import LogUniformProduct
    #     from LPULabelEncoder import LPULabelEncoder
#     from LPULabelDecoder import LPULabelDecoder
    from MultipleVectorizer import MultipleVectorizer
    from RepresentationPacker import RepresentationPacker
    from scorer_library import flexible_scorer
    coo_transformer = TransformToCOO()

    score_func = {'lpu_scorer':flexible_scorer(scorer=metrics.roc_auc_score, average_type='micro', learning_type='lpu'),                                        
                            'private_real_auc_scorer': flexible_scorer(scorer=metrics.roc_auc_score, learning_type='real'),                
                            'private_real_auc_scorer_micro': flexible_scorer(scorer=metrics.roc_auc_score,  average_type='micro', learning_type='real'),                
                            'private_real_auc_scorer_weighted': flexible_scorer(scorer=metrics.roc_auc_score,  average_type='weighted', learning_type='real'),                
                            'private_real_ll_scorer': flexible_scorer(scorer=metrics.log_loss, learning_type='real'),
                            'private_real_aps_scorer': flexible_scorer(scorer=metrics.average_precision_score, learning_type='real'), 
                            'private_real_aps_scorer_micro': flexible_scorer(scorer=metrics.average_precision_score, average_type='micro', learning_type='real'), 
                            'private_real_aps_scorer_weighted': flexible_scorer(scorer=metrics.average_precision_score, average_type='weighted', learning_type='real'), 
                            'private_real_brier_scorer': flexible_scorer(scorer=metrics.brier_score_loss, learning_type='real'), 
                            'private_real_f1_scorer':flexible_scorer(scorer=metrics.f1_score, learning_type='real'),
                            'private_real_f0_scorer':flexible_scorer(scorer=f1_0_score, learning_type='real'),
                            'private_real_elkan_c_scorer':flexible_scorer(scorer=elkan_c_as_score, learning_type='real'), 
                            'private_real_acc_scorer':flexible_scorer(scorer=metrics.accuracy_score, learning_type='real'), 
                            'lpu_aps_scorer':flexible_scorer(scorer=metrics.average_precision_score, learning_type='lpu'),                                                                                  
                            'lpu_aps_scorer_weighted':flexible_scorer(scorer=metrics.average_precision_score, average_type='weighted', learning_type='lpu'),                                                                                  
                            'lpu_aps_scorer_micro':flexible_scorer(scorer=metrics.average_precision_score, average_type='micro', learning_type='lpu'),                                                                                  
                            'lpu_aps_scorer_weighted':flexible_scorer(scorer=metrics.average_precision_score, average_type='weighted', learning_type='lpu'),                                                                                  
                            'lpu_ll_scorer':flexible_scorer(scorer=metrics.log_loss, learning_type='lpu'),                                         
                            'lpu_brier_scorer':flexible_scorer(scorer=metrics.brier_score_loss, learning_type='lpu'),
                            'lpu_auc_scorer':flexible_scorer(scorer=metrics.roc_auc_score, learning_type='lpu'),
                            'lpu_auc_scorer_micro':flexible_scorer(scorer=metrics.roc_auc_score, average_type='micro', learning_type='lpu'),
                            'lpu_auc_scorer_weighted':flexible_scorer(scorer=metrics.roc_auc_score, average_type='weighted', learning_type='lpu'),
                            'lpu_acc_scorer':flexible_scorer(scorer=metrics.accuracy_score, learning_type='lpu'),
                            'lpu_f1_scorer':flexible_scorer(scorer=metrics.f1_score, learning_type='lpu'),                                                          
                            'lpu_f0_scorer':flexible_scorer(scorer=f1_0_score, learning_type='lpu'),     
                            'lpu_elkan_c_scorer':flexible_scorer(scorer=elkan_c_as_score, learning_type='lpu')}
    strat_kfold = StratifiedKFold(5, shuffle=True, random_state=2120)
    pipeline = Pipeline([
                ('packer', RepresentationPacker()),
                ('vectorizer', MultipleVectorizer()),
                ('min_max_scaler', MultiViewMinMaxScaler()),
                ('standard_scaler', MultiViewStandardScaler()),
                ("tocoo", coo_transformer),
                ('clf', PsychM(training_size=len(X_train_new))),
            ])
    
    print ("FUCKING score_func is:", score_func)
#     with parallel_backend('dask'):
    model_cv = RandomizedSearchCV(
                    pipeline,
                    n_iter=n_iter, 
                    scoring=score_func,
                    cv=strat_kfold,
                    param_distributions=param_grid,
                    n_jobs=-1,
                    refit = 'lpu_scorer',
                    error_score='raise'
                )
#     model_cv = RandomizedSearchCV(
#                     pipeline,
#                     n_iter = 5, 
#                     scoring=score_func,
#                     cv=strat_kfold,
#                     param_distributions=param_grid,
#                     n_jobs=-1,
#                     refit = 'lpu_scorer'
#                 )    
    model_cv.fit(X_train_new, l_y_cat_transformed_train)
    return model_cv

# %%
# with parallel_backend('threading', 44) as backend:
#     results=fit_model(X_train_new={'sig_input':sig_X_train, 'psych_input': psych_X_train}, l_y_cat_transformed_train=l_y_cat_transformed_train, param_grid=param_grid)

# %%


# %%
# fit_model(X_train_new=X_train,l_y_cat_transformed_train=l_y_cat_transformed_train, param_grid=param_grid)

# %%
from dask import delayed
from time import time
import sys
sys.path.append('/home/scratch/nshajari/utils')
sys.path.append('/home/scratch/nshajari/LPUModels')
if __name__ == '__main__':

    t = time()
    client.restart()
    # if parallel_type != 'local':
    #     client.restart()
    #     client.upload_file('utils/func_lib.py')
    #     client.upload_file('utils/LogUniformProduct.py')
    #     client.upload_file('utils/IdentityTransformer.py') 
    #     client.upload_file('utils/TransformToCOO.py') 
    #     client.upload_file('utils/MultipleVectorizer.py') 
    #     client.upload_file('utils/RepresentationPacker.py') 
    #     client.upload_file('utils/math_utils.py') 
    #     client.upload_file('utils/tensor_utils.py') 
    #     client.upload_file('LPUModels/MyModel.py') 
    #     client.upload_file('LPUModels/NaiveLPU.py') 
    #     client.upload_file('LPUModels/KMEModel.py') 
    #     client.upload_file('LPUModels/PropensityEM.py') 
    #     client.upload_file('LPUModels/PsychometricLayer.py') 
    #     client.upload_file('LPUModels/GPLayer.py')                 
    #     client.upload_file('LPUModels/PsychMKeras.py') 
    #     client.upload_file('utils/scorer_library.py') 
    #     client.upload_file('LPUModels/PsychM.py') 
    #     client.upload_file('LPUModels/MultiPsychM.py') 
    #     client.upload_file('LPUModels/PUAdapterTF.py') 
    #     client.upload_file('utils/text_utils.py') 
    #     client.upload_file('DataFrameVectorizer.py') 
    #     client.upload_file('bootstrap_library.py') 
    result = delayed(fit_model)(**{'X_train_new':X_train, 'l_y_cat_transformed_train':l_y_cat_transformed_train, 'param_grid':param_grid})
    with parallel_backend('threading') as backend:
#         with dask.config.set({'pool': ThreadPool(88), 
#                        'distributed.worker.multiprocessing-method': 'forkserver',
#                        'distributed.scheduler.work-stealing': True,
#                        'distributed.comm.zstd.threads':  -1,
#                       'distributed.admin.tick.limit':  '100s',
#                       'distributed.scheduler.work-stealing-interval': '1s'

#                       }) as config, parallel_backend("threading") as backend:                     
        results = client.compute(result, sync=True) 
# #     else:
#         client.compute(result,sync=True)
    print ("Processing thsi Randomized CV took", time() - t, "seconds.")

# %% [markdown]
# ### f1_score(y_val, model_cv.best_estimator_['clf'].predict_prob_y_given_x(X_val_new)>0.5)

# %%


# %%
# results.best_estimator_['clf'].final_loss

# %%
# results.best_estimator_['clf'].number_of_successful_attempts = 5

# %%
# results.best_estimator_['clf'].verbose = 0

# %%
# results.best_estimator_['clf'].psych_alpha

# %%
# results.best_estimator_['clf'].fit({'sig_input':X_train[:, :2], 'psych_input': X_train[:, :2]}, l_y_cat_transformed_train)

# %%
# results.best_estimator_['clf']

# %%


# %%
# results.best_estimator_['clf'].all_train_sig_input_features = results.best_estimator_['clf'].all_train_sig_input_features[:, :2]


# %%
# results.best_estimator_['clf'].kernel

# %%
# results.best_estimator_['clf'].manifold_kernel_k

# %%
# results.best_estimator_['clf'].predict_prob_y_given_x(X_val_new).ravel()

# %%
## y_true_stacked = np.stack((1-y_train, y_train), axis=1).reshape((-1, 2))
# y_pred_stacked = np.stack((1-y_pred, y_pred), axis=1).reshape((-1, 2))
# threshold = np.max(y_pred_stacked, axis=-1, keepdims=True)
# # make sure [0, 0, 0] doesn't become [1, 1, 1]
# # Use abs(x) > eps, instead of x != 0 to check for zero
# y_pred_stacked = np.logical_and(y_pred_stacked >= threshold,
#                             tf.abs(y_pred_stacked) > 1e-12)
# print (y_pred_stacked)
# y_true_stacked = y_true_stacked.astype(int)
# y_pred_stacked = y_pred_stacked.astype(int)
# def _count_non_zero(val):
#     non_zeros = np.count_nonzero(val, axis=0)
#     return non_zeros
# result = _count_non_zero(y_pred_stacked * y_true_stacked)
# true_positives = tf.cast(_count_non_zero(y_pred_stacked * y_true_stacked), tf.float32)
# false_positives = tf.cast(_count_non_zero(y_pred_stacked * (y_true_stacked - 1)), tf.float32)
# false_negatives = tf.cast(_count_non_zero((y_pred_stacked - 1) * y_true_stacked), tf.float32)
# precision = tf.math.divide_no_nan(true_positives, (true_positives + false_positives)).numpy()
# recall = tf.math.divide_no_nan(true_positives, (true_positives + false_negatives)).numpy()

# tf.math.divide_no_nan(precision * recall, (precision + recall)) * 

# %%
# transformed_input  = X_train_new
# for transformer_name, transformer in results.best_estimator_.named_steps.items():
#     if transformer_name in ['clf', 'packer']:
#         break
#     transformed_input = transformer.transform(transformed_input)
#     print (transformer_name, transformer.transform)
# #     transformed_input = transformer.transform(transformed_input)
# print (transformed_input)

# %%
#  X_val = np.hstack([X_val, X_val])

# %%
import inspect

func = lambda num1,num2: num1 + num2
funcString = str(inspect.getsourcelines(func)[0])
funcString = funcString.strip("['\\n']").split(" = ")[1]
print (funcString)

# %%
# from scorer_library import safe_brier_score_loss
def safe_brier_score_loss(y_true, y_prob):
    output = y_true - y_prob
    too_small = abs(output) < 1e-8
    output[too_small] = 0.
    return (output ** 2).mean()

# %%


# %%
# copied_best_model_list = [results.best_estimator_] * 10
# def fit_estimator(args):#model):
#     model, X, l_y = args
#     model.fit(X, l_y)
#     # model.fit(transformed_X_train, l_y_cat_transformed_train)
#     return model
# # if 'spm' in 
# parallel_rerunlist = [delayed(fit_estimator)([model, transformed_X_train, l_y_cat_transformed_train]) for model in copied_best_model_list]
# from dask.distributed import get_client
# # client = get_client()


# with parallel_backend('threading') as backend:
#     copied_best_model_results = dask.compute(*parallel_rerunlist)


# #             # client.scatter(transformed_X_train)
# #             # client.scatter(l_y_cat_transformed_train) 
# #             # futures = b.map(fit_estimator).compute()
# # def new_log_loss(l, l_pred):
# #     l_pred = np.nan_to_num(l_pred, nan=0.0)
# #     l_pred = l_pred.astype(np.float64)
# #     # l_pred[l_pred > 1-1e-16] = 1-1e-16
# #     # l_pred[l_pred < 1e-16] = 1e-16
# #     return metrics.log_loss(l, l_pred)
# def score_func(model):
#     if 'psych' in type(model.named_steps['clf']).__name__.lower():
#         print ("***THE ALTERNATING_BEST_LOSS IS being compared****")
#         return model.named_steps['clf'].alternating_best_loss
#     elif 'propensity' in type(model.named_steps['clf']).__name__.lower():
#         print ("The list for SAR-EM is:", model.named_steps['clf'].info['loglikelihoods'])
#         return model.named_steps['clf'].info['loglikelihoods'][-1]
#     else:
#         return -1
# final_list = copied_best_model_results
# # convreged_models_short_list = []
# # for model in copied_best_model_results:
# #     if hasattr(model.named_steps['clf'], 'optimization_results') and model.named_steps['clf'].optimization_results['converged']:
# #         convreged_models_short_list.append(model)
# # if len(convreged_models_short_list):
# # final_list = convreged_models_short_list
# best_model = sorted(final_list, key=lambda x: score_func(x))[0]


# %%
psych_model_results_dict = dict()
# psych_model_results_dict['model_name'] = []
# psych_model_results_dict['test_AUC_score_micro'] = []
# psych_model_results_dict['gp_kernel_amplitude'] = []
# psych_model_results_dict['manifold_kernel_amplitude'] = []
# psych_model_results_dict['gp_kernel_lengthscale'] = []
# psych_model_results_dict['manifold_kernel_lengthscale'] = []
# psych_model_results_dict['lbo_temperature'] = []
# psych_model_results_dict['manifold_kernel_noise'] = []
# psych_model_results_dict['psych_reg_coeff'] = []


for key in results.cv_results_:
    if 'mean' in key and 'scorer' in key:
        psych_model_results_dict[key] = results.cv_results_[key]
    if 'param_clf' in key and ('kernel' in key or 'ratio' in key):
        psych_model_results_dict[key] = results.cv_results_[key]
# pd.DataFrame({'param': clf.cv_results_["params"], 'acc': clf.cv_results_["mean_test_score"]})

#%%
# results.cv_results_.keys()
# for key in psych_model_results_dict.keys():
#     psych_model_results_dict[key].append(np.squeeze(psych_model['CV_estimator'].named_steps['clf'].get_params()[key])                                                                                                        )



# %%
from pandas import DataFrame as pdf
model_dict_df = pdf.from_dict(psych_model_results_dict)
model_dict_df.sort_values(by=['mean_test_private_real_aps_scorer'], inplace=True)
# pd.DataFrame({'param': results.cv_results_["params"], 'acc': results.cv_results_["]})

#%%
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
display(model_dict_df)

#%%
model_dict_df[['mean_test_private_real_auc_scorer_micro', 'param_clf__gp_kernel_type', 'param_clf__gp_kernel_amplitude', 'param_clf__gp_kernel_lengthscale', 'param_clf__manifold_kernel_type', 'param_clf__manifold_kernel_amplitude', 'param_clf__manifold_kernel_lengthscale', 'param_clf__manifold_kernel_noise']]

# %%
from sklearn.metrics import  roc_auc_score, average_precision_score,  accuracy_score
from scorer_library import f1_0_score
from scipy.special import expit
X_experiment = X_val
y_experiment = y_val
l_experiment = l_val
 

transformed_input  = X_experiment
for transformer_name, transformer in results.best_estimator_.named_steps.items():
    print(transformer_name)
    if transformer_name in ['clf']:
        break
    transformed_input = transformer.transform(transformed_input)
#     print (transformer_name, transformer.transform)
#     transformed_input = transformer.transform(transformed_input)
X_experiment = transformed_input
# print(X_experiment)
# print(X_experiment)
y_pred = results.best_estimator_['clf'].predict_prob_y_given_x(X_experiment)
# y_pred = 1 - y_pred
# y_pred = expit(logit(y_pred)+4.28)
# counter = print (y_experiment)
func_dict = {'roc_auc_score':roc_auc_score, 'average_precision_score':average_precision_score, '1-brier_score_loss': lambda x, y: 1 - brier_score_loss(x, y), 'accuracy_score': lambda x, y: accuracy_score(x, y>0.5), 'f1_score': lambda x, y: f1_score(x, y>0.5),  'f0_score':lambda x, y: f1_score(1-x, 1-y>0.5)}
for score_func_name, score_func in func_dict.items():
    print ('y_and_y_pred_errors,', score_func_name,',', score_func(y_experiment, y_pred))
print('-------------------------')
# l_pred = results.best_estimator_['clf'].predict_proba(X_experiment)
l_pred = results.best_estimator_['clf'].predict_prob_y_given_x(X_experiment) * results.best_estimator_['clf'].predict_prob_l_given_y_x(X_experiment)

# l_pred[l_pred<1e-5]=1e-5
# l_pred[l_pred>1-1e-5] = 1-1e-5
for score_func_name, score_func in func_dict.items():
    print('l_and_l_pred_errors,',score_func_name,',', score_func(l_experiment, l_pred))
print('-------------------------')
X_experiment_y_1 = {'sig_input': X_experiment['sig_input'][y_experiment==1], 'psych_input': X_experiment['sig_input'][y_experiment==1]}
l_pred_given_y_1 = results.best_estimator_['clf'].predict_prob_l_given_y_x(X_experiment_y_1)
for score_func_name, score_func in func_dict.items():
    print('l_given_y_1_and_l_pred_given_y_1_errors,',score_func_name,',', score_func(l_experiment[y_experiment==1], l_pred_given_y_1))
print ('-------------------------')
l_pred = results.best_estimator_['clf'].predict_proba(X_experiment)
for score_func_name, score_func in func_dict.items():
    print('y_and_l_pred_errors,', score_func_name,',', score_func(y_experiment, l_pred))


# %%
# for key, value in results.best_estimator_.get_params().items():
#     print (key, value)

# %%
# results.cv_results_

# %%

# %% [markdown]
# #### gamma_lambda_to_g_l_prime_transformer(.2, .2)

# %%
# results.best_estimator_['clf'].parameter_reporter.gamma_hist[0]

# %%
# results.best_estimator_['clf'].parameter_reporter.lambda_hist[0]

# %%
# real_params

# %%
# fig, ax = plt.subplots()
# if hasattr(results.best_estimator_['clf'], 'best_estimator'):
#     best_estimator = results.best_estimator_['clf'].best_estimator_
# else:
#     best_estimator = results.best_estimator_['clf']
    
# # ax.plot(np.asarray(results.best_estimator_['clf'].history['loss']))
# ax.plot(np.asarray(best_estimator.history['lpu_f1_score_for_y']))
# ax.plot(np.asarray(best_estimator.history['lpu_f1_score_for_l']))
# # ax.plot(np.asarray(results.best_estimator_['clf'].parameter_reporter.lambda_hist).reshape((1, -1))[0])

# %%
# fig, ax = plt.subplots()
# ax.plot(np.asarray(results.best_estimator_['clf'].parameter_reporter.gamma_hist).reshape((1, -1))[0])
# ax.plot(np.asarray(results.best_estimator_['clf'].parameter_reporter.lambda_hist).reshape((1, -1))[0])

# %%
results.best_estimator_['clf'].psych_gamma

# %%
from matplotlib import pyplot as plt

# %%
results.best_estimator_['clf'].manifold_kernel_power

# %%
fig, ax = plt.subplots(2, 2)
fig.set_size_inches((10, 5))
ax[0, 0].scatter(sig_X_val[:, 0][y_val==1], sig_X_val[:, 1][y_val==1], alpha=0.8)
ax[0,0].scatter(sig_X_val[:, 0][y_val==0], sig_X_val[:, 1][y_val==0], alpha=0.8)
ax[0, 1].scatter(sig_X_val[:, 0][l_val==1], sig_X_val[:, 1][l_val==1], alpha=0.8)
ax[1, 0].scatter(sig_X_val[:, 0][l_val==0], sig_X_val[:, 1][l_val==0], alpha=0.8)
y_mean = y_pred.mean()
ax[1,1].scatter(sig_X_val[:, 0][y_pred.ravel()<=y_mean], sig_X_val[:, 1][y_pred.ravel()<=y_mean], alpha=0.8)


# %%
keys_list = [
'll_scorer',
'f0_scorer',
'elkan_c_scorer',
'acc_scorer',
'f1_scorer',
'aps_scorer',
'aps_scorer_micro',
'aps_scorer_weighted',
'brier_scorer',
'auc_scorer',
'auc_scorer_weighted',
'auc_scorer_micro',
]



n = 10
real_best_indices_set = set()
lpu_best_indices_set = set()
for key in keys_list:
    if 'mean_test_private_real_'+key not in results.cv_results_:
        print ("The key", key, "is not in the results. Skipping...")
    else:
        print (key)
#         print (sorted(results.cv_results_['rank_test_private_real_'+key]))
        real_n_top = results.cv_results_['rank_test_private_real_'+key].argsort()[:n]#[::-1]
        real_best_indices_set = real_best_indices_set.union(set([results.cv_results_['rank_test_private_real_'+key][item] for item in real_n_top]))
#         print(key, results.cv_results_['mean_test_private_real_'+key][0])
        print ("best key for real", key, "is three indices", real_n_top, "best being", real_n_top[0],  "with value", results.cv_results_['mean_test_private_real_'+key][real_n_top])
#     print(results.cv_results_['mean_test_private_real_'+key][0])
#     print (key, results.cv_results_['mean_test_private_real_'+key][80])
#     print ('real', key, [results.cv_results_['mean_test_private_real_'+key][item] for item in three_top])
#     print (key, ':', min_lpu_idx)
print('----------------------------')
for key in keys_list:
    if 'mean_test_lpu_'+key not in results.cv_results_:
        print ("The key", key, "is not in the results. Skipping...")
    else:
        lpu_n_top = results.cv_results_['rank_test_lpu_'+key].argsort()[:n]#[::-1]
        lpu_best_indices_set = lpu_best_indices_set.union(set([results.cv_results_['rank_test_lpu_'+key][item] for item in lpu_n_top]))
#         print(key, results.cv_results_['mean_test_private_real_'+key][0])
        print ("best key for lpu", key, "is three indices", lpu_n_top, "best being", lpu_n_top[0],  "with value", results.cv_results_['mean_test_lpu_'+key][lpu_n_top])
#     print (key, results.cv_results_['mean_test_private_lpu_'+key][80])
#     print ('real', key, [results.cv_results_['mean_test_private_lpu_'+key][item] for item in [1, 3, 8, 16, 19, 21, 24, 26, 28, 29, 30, 31, 32, 33, 35, 37, 45, 47, 53, 59, 61, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 79, 80, 90, 93, 99]])
print(real_best_indices_set)
print(lpu_best_indices_set)
print(lpu_best_indices_set.intersection(real_best_indices_set))


    # results.cv_resuls_

# %%
real_best_indices_set

# %%
from copy import  deepcopy
holder_estimator = deepcopy(results.best_estimator_)

# %%
lpu_n_top

# %%
# for key in results.cv_results_.keys():
#     if 'mean' in key and key and 'test' in key:
#         print(key, results.cv_results_[key][17])

# %%
import seaborn as sns
sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7)
# %matplotlib inline

# %%
# results.cv_results_['params'][item]

# %%
from copy import deepcopy

# %%
import matplotlib.pyplot as plt
import numpy as np
holder_estimators_dict = dict()
for iter_num, item in enumerate(lpu_n_top):
    print ("**************** PROCESSING ITEM NO **************** :", iter_num, "which is item:", item)
    holder_estimator = deepcopy(holder_estimator)
    holder_estimator.set_params(**results.cv_results_['params'][item])
    holder_estimator.fit(X_train, l_y_cat_transformed_train)
    holder_estimators_dict[item] = holder_estimator
    for key in results.cv_results_.keys():
        if 'mean' in key and key and 'test' in key:
            print(key, results.cv_results_[key][item])
    # ax = fig.gca(projection='3d')

    # print (model['model_name'], "manifold_kernel_power is:", manifold_kernel_power)

    # print (sig_X)



# %%
# holder_estimators_dict = [bad_model['CV_estimator']] * 5

# %%


# %%
# lpu_n_top = range(5)
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from scorer_library import elkan_c_as_score, f1_0_score

# %%
fig, ax = plt.subplots(len(lpu_n_top), 2)
fig.set_size_inches(10, len(lpu_n_top) * 5)
import matplotlib.pyplot as plt
import numpy as np

for iter_num, item in enumerate(lpu_n_top):
    holder_estimator = holder_estimators_dict[item]
    
    X_min = np.min(sig_X[:, 0]) * 3
    X_max = np.max(sig_X[:, 0]) * 3
    Y_min = np.min(sig_X[:, 1]) * 3
    Y_max = np.max(sig_X[:, 1]) * 3
    X_grid = np.arange(X_min, X_max, 0.1)
    Y_grid = np.arange(Y_min, Y_max, 0.1)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    p_y_x_Z = []
    p_l_y_x_Z = []
    true_p_l_y_x_Z = []
    print ("sig_a is:", holder_estimator.named_steps['clf'].sig_a[:5])
    
    # p_l_x_Z[]
    temp_input = dict()
    a = 0
    b = 0
    for x, y in zip(X_grid, Y_grid):
        sig_input = np.hstack([x.reshape((-1, 1)) -a, y.reshape((-1, 1))-b])
        psych_input = np.hstack([x.reshape((-1, 1))-a, y.reshape((-1, 1))-b])
        transformed_input  = np.hstack((sig_input, psych_input))
        for transformer_name, transformer in holder_estimator.named_steps.items():
            if transformer_name in ['clf']:
                continue
            transformed_input = transformer.transform(transformed_input)
        temp_input = transformed_input
        # print ("OH NO!!!! sig_a should be :", results.cv_results_['params'][item]['sig_a'][:5])
        p_y_x_Z.append(holder_estimator.named_steps['clf'].predict_prob_y_given_x(temp_input))
        try:
            p_l_y_x_Z.append(holder_estimator.named_steps['clf'].predict_prob_l_given_y_x(temp_input))
        except FloatingPointError:
            p_l_y_x_Z.append(np.zeros_like(true_alpha @ temp_input['psych_input'].T + true_beta).reshape((-1,1)))
        true_p_l_y_x_Z.append(true_gamma + (1-true_gamma-true_lambda) * tf.sigmoid(true_alpha @ temp_input['psych_input'].T + true_beta).numpy())
    #     p_l_y_x_Z.append(results.best_estimator_['clf'].predict_proba(temp_input))
    p_y_x_Z = np.asarray(p_y_x_Z).reshape(X_grid.shape)
    p_l_y_x_Z = np.asarray(p_l_y_x_Z).reshape(X_grid.shape)
    sigma_I = np.round(np.squeeze(holder_estimator.named_steps['clf'].manifold_kernel_lengthscale), 3)
    gamma_I = np.round(np.squeeze(holder_estimator.named_steps['clf'].manifold_kernel_amplitude), 3)
    sigma_A = np.round(np.squeeze(holder_estimator.named_steps['clf'].gp_kernel_lengthscale), 3)
    gamma_A = np.round(np.squeeze(holder_estimator.named_steps['clf'].gp_kernel_amplitude), 3)
    gamma_A_I_ratio = gamma_A / gamma_I
    connectivity = holder_estimator.named_steps['clf'].manifold_neighbor_mode
    manifold_kernel_k = holder_estimator.named_steps['clf'].manifold_kernel_k
    manifold_kernel_noise = np.round(holder_estimator.named_steps['clf'].manifold_kernel_noise, 3)
    lbo_temperature = np.round(holder_estimator.named_steps['clf'].lbo_temperature, 3)
    converged = holder_estimator.named_steps['clf'].optimization_results['converged'].numpy()
    manifold_kernel_normed = holder_estimator.named_steps['clf'].manifold_kernel_normed
    manifold_kernel_power = np.round(holder_estimator.named_steps['clf'].manifold_kernel_power, 3)
    is_SPM = holder_estimator.named_steps['clf'].is_SPM
    
    ax[iter_num, 0].contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=50)
    ax[iter_num, 1].contourf(X_grid, Y_grid, true_p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=10)
    ax[iter_num, 0].set_title("model_idx:" + str(item)+ r'$, \gamma_I=$'+str(gamma_I)+r'$, \sigma_I=$'+str(sigma_I)+r'$, \gamma_A/\gamma_I=$'+str(gamma_A_I_ratio) + r'$, \gamma_A=$'+str(gamma_A)+r'$, \sigma_A=$'+str(sigma_A) + '\n'+', cnctvty: '+connectivity + r'$, \nu$='+str(manifold_kernel_noise)+r'$, T_{LBO}=$'+str(lbo_temperature)+r'$, k=$'+str(manifold_kernel_k)+', converged:'+str(converged)+'\n'+'manifold kernel normed:'+str(manifold_kernel_normed)+', manifold kernel power:'+str(manifold_kernel_power) + ', ROC AUC SCORE:' +str(roc_auc_score)+'\n'+\
                                     r'$, \gamma=$' + str(true_gamma) +\
                                    r'$, \lambda=$' + str(true_lambda) +\
                                    r'$, \alpha=$' + str(true_alpha)+\
                                    r'$, \beta=$' + str(true_beta)+\
                                    'is SPM?' + str(is_SPM) + ', \#l:' + str(l_train.sum()) + ', l\%:' + str(l_train.shape[0]))
    # ax.contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=20)
    # ax.scatter(original_model_dict['sig_X_train'][real_l_train.astype(bool), 0], original_model_dict['sig_X_train'][real_l_train.astype(bool), 1],alpha=0.6, c='k', marker='+')
    # ax.scatter(original_model_dict['sig_X_train'][~(real_l_train.astype(bool)), 0], original_model_dict['sig_X_train'][~(real_l_train.astype(bool)), 1],alpha=0.6, c='g', marker='$?$')
    # ax[iter_num, 0].scatter(X_train[~(l_train.astype(bool)), 0], X_train[~(l_train.astype(bool)), 1],alpha=0.6, c='g', marker='$?$')
    ax[iter_num, 0].scatter(X_train[l_train.astype(bool), 0], X_train[l_train.astype(bool), 1],alpha=0.6, c='k', marker='+')
    # res = ax[0].scatter(X_grid, Y_grid, c=real_p_l_y_x_contour_mesh, cmap=plt.cm.PuOr, alpha=0.9, s=5)#, levels=10)#, extend='both')
    res = ax[iter_num, 1].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.BrBG, alpha=0.4, extend='both', levels=5)
    # res = ax.contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.BrBG, alpha=0.4, extend='both', levels=5)
fig.tight_layout()

# %%
best_estimator = holder_estimators_dict[lpu_n_top[0]]

# %%
transformed_X_test = hstack([sig_X_test,  psych_X_test])
transformed_X_train = hstack([sig_X_train,  psych_X_train])


# %%
# client.shutdown()
# client = LocalCluster()
# client.restart()

# %%
# copied_best_model_list = [best_estimator] * 5
# for model in copied_best_model_list:
#     model.params_sample = None
# if __name__ == '__main__':
#     def fit_estimator(args):#model):
#         model, X, l_y = args
#         model.fit(X, l_y)
#         # model.fit(transformed_X_train, l_y_cat_transformed_train)
#         return model
#     # if 'spm' in 
#     copied_best_model_results = [fit_estimator([model, transformed_X_train, l_y_cat_transformed_train]) for model in copied_best_model_list]
#     # from dask.distributed import get_client
#     # client = get_client()
#     # client.restart()
#     # import dill
#     # dill.extend(False)
#     # with parallel_backend('dask') as backend:
#         # copied_best_model_results = dask.compute(*parallel_rerunlist)


#     #             # client.scatter(transformed_X_train)
#     #             # client.scatter(l_y_cat_transformed_train) 
#     #             # futures = b.map(fit_estimator).compute()
#     # def new_log_loss(l, l_pred):
#     #     l_pred = np.nan_to_num(l_pred, nan=0.0)
#     #     l_pred = l_pred.astype(np.float64)
#     #     # l_pred[l_pred > 1-1e-16] = 1-1e-16
#     #     # l_pred[l_pred < 1e-16] = 1e-16
#     #     return metrics.log_loss(l, l_pred)
#     def score_func(model):
#         if 'psych' in type(model.named_steps['clf']).__name__.lower():
#             print ("***THE ALTERNATING_BEST_LOSS IS being compared****")
#             return model.named_steps['clf'].alternating_best_loss
#         elif 'propensity' in type(model.named_steps['clf']).__name__.lower():
#             print ("The list for SAR-EM is:", model.named_steps['clf'].info['loglikelihoods'])
#             return model.named_steps['clf'].info['loglikelihoods'][-1]
#         else:
#             return -1
#     final_list = copied_best_model_results
#     # convreged_models_short_list = []
#     # for model in copied_best_model_results:
#     #     if hasattr(model.named_steps['clf'], 'optimization_results') and model.named_steps['clf'].optimization_results['converged']:
#     #         convreged_models_short_list.append(model)
#     # if len(convreged_models_short_list):
#     # final_list = convreged_models_short_list
#     best_model = sorted(final_list, key=lambda x: score_func(x))[0]

# %%
fig, ax = plt.subplots(1)
fig.set_size_inches(5,5)
X_min = np.min(sig_X[:, 0]) * 3
X_max = np.max(sig_X[:, 0]) * 3
Y_min = np.min(sig_X[:, 1]) * 3
Y_max = np.max(sig_X[:, 1]) * 3
X_grid = np.arange(X_min, X_max, 0.1)
Y_grid = np.arange(Y_min, Y_max, 0.1)
X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
p_y_x_Z = []
p_l_y_x_Z = []
true_p_l_y_x_Z = []
print ("sig_a is:", best_model.named_steps['clf'].sig_a[:5])

# p_l_x_Z[]
temp_input = dict()
a = 0
b = 0
for x, y in zip(X_grid, Y_grid):
    sig_input = np.hstack([x.reshape((-1, 1)) -a, y.reshape((-1, 1))-b])
    psych_input = np.hstack([x.reshape((-1, 1))-a, y.reshape((-1, 1))-b])
    transformed_input  = np.hstack((sig_input, psych_input))
    for transformer_name, transformer in best_model.named_steps.items():
        if transformer_name in ['clf']:
            continue
        transformed_input = transformer.transform(transformed_input)
    temp_input = transformed_input
    # print ("OH NO!!!! sig_a should be :", results.cv_results_['params'][item]['sig_a'][:5])
    p_y_x_Z.append(best_model.named_steps['clf'].predict_prob_y_given_x(temp_input))
    p_l_y_x_Z.append(best_model.named_steps['clf'].predict_prob_l_given_y_x(temp_input))
    true_p_l_y_x_Z.append(true_gamma + (1-true_gamma-true_lambda) * expit(true_alpha @ temp_input['psych_input'].T) + true_beta)
#     p_l_y_x_Z.append(results.best_estimator_['clf'].predict_proba(temp_input))
p_y_x_Z = np.asarray(p_y_x_Z).reshape(X_grid.shape)
p_l_y_x_Z = np.asarray(p_l_y_x_Z).reshape(X_grid.shape)
sigma_I = np.round(np.squeeze(best_model.named_steps['clf'].manifold_kernel_lengthscale), 3)
gamma_I = np.round(np.squeeze(best_model.named_steps['clf'].manifold_kernel_amplitude), 3)
sigma_A = np.round(np.squeeze(best_model.named_steps['clf'].gp_kernel_lengthscale), 3)
gamma_A = np.round(np.squeeze(best_model.named_steps['clf'].gp_kernel_amplitude), 3)
gamma_A_I_ratio = gamma_A / gamma_I
connectivity = best_model.named_steps['clf'].manifold_neighbor_mode
manifold_kernel_k = best_model.named_steps['clf'].manifold_kernel_k
manifold_kernel_noise = np.round(best_model.named_steps['clf'].manifold_kernel_noise, 3)
lbo_temperature = np.round(best_model.named_steps['clf'].lbo_temperature, 3)
converged = best_model.named_steps['clf'].optimization_results['converged'].numpy()
manifold_kernel_normed = best_model.named_steps['clf'].manifold_kernel_normed
manifold_kernel_power = np.round(best_model.named_steps['clf'].manifold_kernel_power, 3)
is_SPM = best_model.named_steps['clf'].is_SPM

ax.contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=50)
ax.contourf(X_grid, Y_grid, true_p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=10)
ax.set_title("model_idx:" + str(item)+ r'$, \gamma_I=$'+str(gamma_I)+r'$, \sigma_I=$'+str(sigma_I)+r'$, \gamma_A/\gamma_I=$'+str(gamma_A_I_ratio) + r'$, \gamma_A=$'+str(gamma_A)+r'$, \sigma_A=$'+str(sigma_A) + '\n'+', cnctvty: '+connectivity + r'$, \nu$='+str(manifold_kernel_noise)+r'$, T_{LBO}=$'+str(lbo_temperature)+r'$, k=$'+str(manifold_kernel_k)+', converged:'+str(converged)+'\n'+'manifold kernel normed:'+str(manifold_kernel_normed)+', manifold kernel power:'+str(manifold_kernel_power) + ', ROC AUC SCORE:' +str(roc_auc_score)+'\n'+\
                                 r'$, \gamma=$' + str(true_gamma) +\
                                r'$, \lambda=$' + str(true_lambda) +\
                                r'$, \alpha=$' + str(true_alpha)+\
                                r'$, \beta=$' + str(true_beta)+\
                                'is SPM?' + str(is_SPM) + ', \#l:' + str(l_train.sum()) + ', l\%:' + str(l_train.shape[0]))
# ax.contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3, extend='both', levels=20)
# ax.scatter(original_model_dict['sig_X_train'][real_l_train.astype(bool), 0], original_model_dict['sig_X_train'][real_l_train.astype(bool), 1],alpha=0.6, c='k', marker='+')
# ax.scatter(original_model_dict['sig_X_train'][~(real_l_train.astype(bool)), 0], original_model_dict['sig_X_train'][~(real_l_train.astype(bool)), 1],alpha=0.6, c='g', marker='$?$')
# ax[iter_num, 0].scatter(X_train[~(l_train.astype(bool)), 0], X_train[~(l_train.astype(bool)), 1],alpha=0.6, c='g', marker='$?$')
ax.scatter(X_train[l_train.astype(bool), 0], X_train[l_train.astype(bool), 1],alpha=0.6, c='k', marker='+')
ax.scatter(X_train[~y_train.astype(bool), 0], X_train[~y_train.astype(bool), 1],alpha=0.6, c='r', marker='$?$')
# res = ax[0].scatter(X_grid, Y_grid, c=real_p_l_y_x_contour_mesh, cmap=plt.cm.PuOr, alpha=0.9, s=5)#, levels=10)#, extend='both')
# res = ax.contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.BrBG, alpha=0.4, extend='both', levels=5)

# %%
transformed_X_train = np.hstack([sig_X, sig_X])

for key, _ in holder_estimator.steps:
    if key == 'clf':
        break
    transformed_X_train = holder_estimator[key].transform(transformed_X_train)
    # counter += 1


# %%


# %%
fig, ax = plt.subplots(1)
fig.set_size_inches(15, 10)
plt.hist(holder_estimator.predict_proba(X_train), color='darkorange', alpha=0.5, density=True)
clf_out = holder_estimator.named_steps['clf'].predict_prob_y_given_x(transformed_X_train)
psych_out = holder_estimator.named_steps['clf'].predict_prob_l_given_y_x(transformed_X_train)
ax.hist(clf_out, color='darkblue', alpha=0.5, density=True)
ax.hist(psych_out, color='darkred', alpha=0.5, density=True)
ax.hist(psych_out * clf_out, color='darkgreen', alpha=0.5, density=True)



# %%
holder_estimator.named_steps['clf'].optimizer_dict

# %%
holder_estimator.named_steps['clf'].alternating_best_loss

# %%
holder_estimator.named_steps['clf'].manifold_kernel_amplitude

# %%
holder_estimator.named_steps['clf'].gp_kernel_lengthscale

# %%
holder_estimator.named_steps['clf'].gp_kernel_amplitude

# %%


# %%
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



# p_l_x_Z = np.asarray(p_l_x_Z).reshape(X.shape)
# print (Z)
# print(results.best_estimator_['clf'].predict_prob_l_given_y_x(X_val_new))
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# #Plot the surface.
# surf_p_y_x = ax.plot_surface(X_grid, Y_grid, p_y_x_Z,
#                        linewidth=1, antialiased=True, alpha=0.6, cmap=cm.coolwarm)
# surf_p_l_y_x = ax.plot_surface(X_grid, Y_grid, p_l_y_x_Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.scatter(sig_X_train[:, 0][l_train==0], sig_X_train[:, 1][l_train==0],alpha=.5, color='k', marker='$?$')
# ax.scatter(sig_X_train[:, 0][l_train==1], sig_X_train[:, 1][l_train==1],alpha=.5, color='r', marker='+')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf_p_y_x, shrink=0.5, aspect=5)
# # fig.colorbar(surf_p_l_y_x, shrink=0.5, aspect=5)

# plt.show()


# %%
sig_X_train.shape

# %%
# ax.scatter(sig_X_train[:, 0], sig_X_train[:, 1], )
# plt.show()

# %%
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
ax[0].contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.6, levels=40)
ax[1].contourf(X_grid, 
                  Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=.8, levels=40)
ax[0].contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.6, levels=40)
# ax.scatter(X, Y, p_y_x_Z, cmap=plt.cm.RdYlBu)
# ax.scatter(sig_X_val[:, 0][y_val==1], sig_X_val[:, 1][y_val==1], alpha=0.8,c='b', cmap=plt.cm.RdYlBu, marker='+')
# ax.scatter(sig_X_val[:, 0][y_val==0], sig_X_val[:, 1][y_val==0], alpha=0.8,c='r', cmap=plt.cm.RdYlBu,  marker='_')
# ax[0].scatter(sig_X_train[:, 0][l_train==0], sig_X_train[:, 1][l_train==0], alpha=0.8,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
# ax[0].scatter(sig_X_train[:, 0][l_train==1], sig_X_train[:, 1][l_train==1], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
ax[0].set_xlabel(r'$X_1$')
ax[0].set_ylabel(r'$X_2$')
ax[0].set_title(r'Training data')
# ax[1].scatter(sig_X_val[:, 0][l_val==0], sig_X_val[:, 1][l_val==0], alpha=0.8,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
# ax[1].scatter(sig_X_val[:, 0][l_val==1], sig_X_val[:, 1][l_val==1], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
ax[1].set_xlabel(r'$X_1$')
ax[1].set_ylabel(r'$X_2$')
ax[1].set_title(r'Test data')
# ax[1, 0].scatter(sig_X[:, 0][l==0], sig_X[:, 1][l==0], alpha=0.8,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
# ax[1, 0].scatter(sig_X[:, 0][l==1], sig_X[:, 1][l==1], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
# ax[1, 0].set_xlabel(r'$X_1$')
# ax[1, 0].set_ylabel(r'$X_2$')
# ax[1, 0].set_title(r'All data')
# ax[1, 1].spines['top'].set_visible(False)
# ax[1, 1].spines['right'].set_visible(False)
# ax[1, 1].spines['left'].set_visible(False)
# ax[1, 1].spines['bottom'].set_visible(False)
# ax[1, 1].set_xticks([])
# ax[1, 1].set_yticks([])
# for ax_ in ax.flatten():
#     print (ax_)
    # ax_.set_aspect('1')
# fig.tight_layout()

# fig.close()



# %%
fig, ax = plt.subplots()
ax.contourf(X_grid, Y_grid, p_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.6, levels=40)
ax.scatter(sig_X_train[:, 0][l_train==0], sig_X_train[:, 1][l_train==0], alpha=0.8,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
ax.scatter(sig_X_train[:, 0][l_train==1], sig_X_train[:, 1][l_train==1], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_aspect('1')
# plt.axis('off')
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False, 
    right=False,
    left=False,# ticks along the top edge are off
    labelbottom=False,
labelleft=False) # labels along the bottom edge are off
fig.savefig(X_y_sample_type+'_prediction.pdf', transparent=True)


# %%
results.best_estimator_['clf'].g_prime

# %%
results.best_estimator_['clf'].psych_gamma

# %%
results.best_estimator_['clf'].psych_lambda

# %%
results.best_estimator_['clf'].sig_b

# %%
results.best_estimator_['clf'].optimization_results

# %%
results.best_estimator_['clf'].psych_beta

# %%
fig, ax = plt.subplots()
ax.hist(results.best_estimator_['clf'].sig_a, bins=50)

# %%
# for key, value in results.cv_results_.items():
#     if 'score' in key or 'rank' in key:
#         print (key, value)

# %%
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(10, 10)
ax[0, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
ax[0, 1].contourf(X_grid, 
                  Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
ax[1, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
# ax.scatter(X, Y, p_y_x_Z, cmap=plt.cm.RdYlBu)
# ax.scatter(sig_X_val[:, 0][y_val==1], sig_X_val[:, 1][y_val==1], alpha=0.8,c='b', cmap=plt.cm.RdYlBu, marker='+')
# ax.scatter(sig_X_val[:, 0][y_val==0], sig_X_val[:, 1][y_val==0], alpha=0.8,c='r', cmap=plt.cm.RdYlBu,  marker='_')
ax[0, 0].scatter(sig_X_train[:, 0][np.logical_and(l_train==0, y_train==1)] , sig_X_train[:, 1][np.logical_and(l_train==0, y_train==1)], alpha=0.6,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
ax[0, 0].scatter(sig_X_train[:, 0][np.logical_and(l_train==1, y_train==1)], sig_X_train[:, 1][np.logical_and(l_train==1, y_train==1)], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
ax[0, 0].set_xlabel(r'$X_1$')
ax[0, 0].set_ylabel(r'$X_2$')
ax[0, 0].set_title(r'Training data')
ax[0, 1].scatter(sig_X_val[:, 0][np.logical_and(l_val==0, y_val==1)], sig_X_val[:, 1][np.logical_and(l_val==0, y_val==1)], alpha=0.6,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
ax[0, 1].scatter(sig_X_val[:, 0][np.logical_and(l_val==1, y_val==1)], sig_X_val[:, 1][np.logical_and(l_val==1, y_val==1)], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
ax[0, 1].set_xlabel(r'$X_1$')
ax[0, 1].set_ylabel(r'$X_2$')
ax[0, 1].set_title(r'Test data')
ax[1, 0].scatter(X_dict['sig_input'][:, 0][np.logical_and(l==0, sig_y==1)], X_dict['sig_input'][:, 1][np.logical_and(l==0, sig_y==1)], alpha=0.6,c='k', cmap=plt.cm.RdYlBu, marker=r'$?$', lw=0.1)
ax[1, 0].scatter(X_dict['sig_input'][:, 0][np.logical_and(l==1, sig_y==1)], X_dict['sig_input'][:, 1][np.logical_and(l==1, sig_y==1)], alpha=0.6,c='b', cmap=plt.cm.RdYlBu, marker=r'$+$', lw=0.1)
ax[1, 0].set_xlabel(r'$X_1$')
ax[1, 0].set_ylabel(r'$X_2$')
ax[1, 0].set_title(r'All data')
# fig.tight_layout()
for ax_ in ax.flatten():
#     print (ax_)
    ax_.set_aspect('1')
ax[1, 1].spines['top'].set_visible(False)
ax[1, 1].spines['right'].set_visible(False)
ax[1, 1].spines['left'].set_visible(False)
ax[1, 1].spines['bottom'].set_visible(False)
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

# fig.close()



# %%
# results.best_estimator_['clf'].parameter_reporter.lap_lengthscale_hist

# %%
fig, ax = plt.subplots()
ax.plot(results.best_estimator_['clf'].history['loss'] - np.min(results.best_estimator_['clf'].history['loss']) + .1)
ax.set_yscale('log')
# ax.set_ylim(-1, 100000000000)

# %%
results.best_estimator_['clf'].parameter_reporter.gp_lengthscale_hist

# %%
fig, ax = plt.subplots()
ax.plot(np.asarray(results.best_estimator_['clf'].parameter_reporter.gp_lengthscale_hist).squeeze())
# ax.set_yscale('log')

# %%
results.best_estimator_['clf'].keras_model

# %%
print (X_grid[0, 0], Y_grid[0, 0])
print (X_grid[-1, -1], Y_grid[-1, -1])



# %%
model = results.best_estimator_['clf']

# %%
model.initialize_model()

# %%
y_pred = model.keras_model({'sig_input': sig_X_train, 'psych_input':psych_X_train, 'idx':[]})

# %%
y_pred

# %%
batch_size = len(l_y_cat_transformed_train)

# %%
import tensorflow as tf

# %%
sig_layer_input_shape = len(l_y_cat_transformed_train)
psych_layer_input_shape = 2

# %%
g_prime, l_prime, psych_linear, psych_alpha, psych_beta,\
sig_linear, sig_a, sig_b = \
tf.cast(y_pred[0, 0], tf.float64), tf.cast(y_pred[1, 0], tf.float64), tf.cast(y_pred[2: batch_size + 2, :], tf.float64), tf.cast(y_pred[batch_size + 2:psych_layer_input_shape + batch_size + 2, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + batch_size + 2, :], tf.float64),\
tf.cast(y_pred[psych_layer_input_shape + batch_size + 3:psych_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + batch_size * 2 + 3: psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64), tf.cast(y_pred[psych_layer_input_shape + sig_layer_input_shape + batch_size * 2 + 3, :], tf.float64)      


# %%
from PsychMKeras import expanded_using_softmax

# %%
l_true = l_train.astype(float)

# %%


# %%
# l_true.values

# %%
@tf.function
def expanded_using_softmax(l, g_prime, l_prime, psych_linear, sig_linear):
    L_s = psych_linear
    L_t = sig_linear
    log_sum_exp = tf.math.reduce_logsumexp
    
    # NOTICE: 
    # C is not appearing because C = E - D + F
    # Also A = log(D_star) + G + H
    # Also C = E - D + F
    try:
        shape = tf.shape(L_s)
        tf.print ("shape:", shape)
        tf.print("L_s:", L_s)
        zero = tf.constant(0., tf.float64)
        tf.print ("secod element shape:", log_sum_exp(tf.concat([-tf.tile([[g_prime]], shape), -g_prime - L_s,  tf.tile([[zero]], shape)], axis=1), axis=1, name='A_leftover').shape)
        tf.print ("first element shape:", l.shape)
        
        A_leftover = tf.multiply(l, log_sum_exp(tf.concat([-tf.tile([[g_prime]], shape), -g_prime - L_s,  tf.tile([[zero]], shape)], axis=1), axis=1, name='A_leftover'))
        tf.print ("A_leftover:", A_leftover)
        H_F_B_G =  - log_sum_exp(tf.concat([tf.tile([[zero]], shape), -L_s], axis=1), axis=1, name='H_F_B_G_1') - log_sum_exp(tf.concat([tf.tile([[zero]], shape), -L_t], axis=1), axis=1, name='H_F_B_G_2') \
        - log_sum_exp(tf.concat([tf.tile([[zero]], shape), tf.tile([[-l_prime]], shape), tf.tile([[-g_prime]],shape)], axis=1), axis=1, name='H_F_B_G_3')
        E_D = tf.multiply(1-l, log_sum_exp(tf.concat([tf.tile([[-l_prime]], shape), -L_s, -L_s - l_prime, -L_t, -L_t - g_prime, -L_t - l_prime, -L_s - L_t, -L_s - L_t - g_prime, -L_s - L_t - l_prime], axis=1), axis=1, name='E_D'))
        log_like = A_leftover + H_F_B_G + E_D
    except Exception as e:
        raise type(e)(str(e) + "error is happening in expanded_using_softmax")
    
    return log_like

# %%
l_true.values

# %%
expanded_using_softmax(l_true.values.reshape((-1, 1)), g_prime, l_prime, psych_linear, sig_linear)

# %%
# results.best_estimator_['clf'].best_estimator_.psych_lambda_

# %%
real_params

# %%
# fig, ax = plt.subplots(2)
# ax[0].plot(results.best_estimator_['clf'].best_estimator_.history['loss'])
# ax[1].plot(results.best_estimator_['clf'].best_estimator_.history['lpu_f1_score_for_y'])
# results.best_estimator_['clf'].best_estimator_.history['loss'][-1]

# %%
print(results.best_estimator_['clf'].psych_alpha, results.best_estimator_['clf'].psych_beta)
print(results.best_estimator_['clf'].psych_gamma, results.best_estimator_['clf'].psych_lambda)


# %%
results.best_estimator_['clf'].best_estimator_.final_loss

# %%
results.best_estimator_['clf'].best_estimator_.final_success

# %%
-4.98638837/0.00179243

# %%
0.04301212/0.0001848

# %%
from matplotlib import pyplot as plt
# %matplotlib widget
fig, ax = plt.subplots()
ax.plot(np.asarray(stored_psych_model[2]['lpu_f1_score_for_y'])[:, 0], label='y_f1_score+')
ax.plot(np.asarray(stored_psych_model[2]['lpu_f1_score_for_y'])[:, 1], label='y_f1_score-')
ax.plot(np.asarray(stored_psych_model[2]['lpu_f1_score_for_l'])[:, 0], label='l_f1_score+')
ax.plot(np.asarray(stored_psych_model[2]['lpu_f1_score_for_l'])[:, 1], label='l_f1_score-')
ax.plot(np.asarray(stored_psych_model[2]['lpu_brier_score_for_y']), label='y_brier_score')
ax.plot(np.asarray(stored_psych_model[2]['lpu_brier_score_for_l']), label='l_brier_score')

ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.plot(np.asarray(stored_psych_model[2]['loss']), label='loss')
stored_psych_model[2]['lpu_f1_score_for_y']
fig.legend()

# %%
results.best_estimator_['clf'].history

from matplotlib import pyplot as plt
# %matplotlib widget
fig, ax = plt.subplots()
best_estimator_history = results.best_estimator_['clf'].history
best_estimator_history
ax.plot(np.asarray(best_estimator_history['lpu_f1_score_for_y'])[:, 0], label='y_f1_score+')
ax.plot(np.asarray(best_estimator_history['lpu_f1_score_for_y'])[:, 1], label='y_f1_score-')
ax.plot(np.asarray(best_estimator_history['lpu_f1_score_for_l'])[:, 0], label='l_f1_score+')
ax.plot(np.asarray(best_estimator_history['lpu_f1_score_for_l'])[:, 1], label='l_f1_score-')

ax.plot(np.asarray(best_estimator_history['lpu_brier_score_for_y']), label='y_brier_score')
ax.plot(np.asarray(best_estimator_history['lpu_brier_score_for_l']), label='l_brier_score')

ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.plot(np.asarray(best_estimator_history['loss']), label='loss')
best_estimator_history['lpu_f1_score_for_y']
fig.legend()

# %%


# %%
from scorer_library import scorer_general

# %%


# %%


# %%
test_model.set_params(**stored_psych_model[4])

# %%
from sklearn.metrics import f1_score

# %%


# %%
# test_model.all_train_sig_input_features = stored_psych_model[0]

# %%
import copy

# %%
test_pipeline = copy.deepcopy(results.best_estimator_)

# %%
test_pipeline.steps[5] = ('clf', test_model)

# %%
test_model.kernel

# %%
scorer_general(test_pipeline, np.hstack([stored_psych_model[0]['sig_input'], stored_psych_model[0]['sig_input']]), test_model.encoder.transform(stored_psych_model[1]), scorer=f1_score, learning_type='real')

# %%
scorer_general(test_pipeline, np.hstack([stored_psych_model[0]['sig_input'], stored_psych_model[0]['sig_input']]), test_model.encoder.transform(stored_psych_model[1]), scorer=brier_score_loss, learning_type='real')



# %%
test_pipeline['clf'].predict_proba(stored_psych_model[0]).shape

# %%


# %%
a = stored_keras_model.sig_layer({'sig_input':tf.convert_to_tensor(stored_psych_model[0]['sig_input'], dtype=tf.float64), 'idx': tf.convert_to_tensor([[1.]])}, training=False)
b = stored_keras_model.psych_layer({'psych_input':tf.convert_to_tensor(stored_psych_model[0]['sig_input'], dtype=tf.float64), 'idx': tf.convert_to_tensor([[1.]])}, training=False)
c = tf.sigmoid(a[2]).numpy() 
temp_gamma = abs(b[1]) / (abs(b[1]) + abs(b[2]) + 1)
temp_lambda = abs(b[2]) / (abs(b[1]) + abs(b[2]) + 1)
print (temp_gamma, temp_lambda)
d = (tf.sigmoid(b[0]) * (1 - temp_gamma - temp_lambda) + temp_gamma).numpy()
# d[d < 1e-100] = 0.
# c[c< 1e-100] = 0. 
# print("HI:", tf.reduce_mean(tf.square(current_l.reshape((-1, 1)) - tf.multiply(c, d))))
# print (c)
# ((c - current_l.reshape((-1, 1))) ** 2).mean()

# stored_keras_model({'sig_input':tf.convert_to_tensor(stored_psych_model[0]['sig_input'], dtype=tf.float64), 'idx': tf.convert_to_tensor([[1.]]), 'psych_input':tf.convert_to_tensor(stored_psych_model[0]['sig_input'])}, training=False).numpy().shape# - current_l.reshape((-1, 1))



# test_pipeline['clf'].predict_proba(stored_psych_model[0]).shape


# %%


# %%
np.min(stored_psych_model[2]['lpu_brier_score_for_l'])

# %%
stored_psych_model[0]['sig_input'].shape

# %%
# scorer_general(test_pipeline, np.hstack([stored_psych_model[0]['sig_input'], stored_psych_model[0]['sig_input']]), test_model.encoder.transform(stored_psych_model[1]), scorer=brier_score_loss, learning_type='lpu')


# %%
scorer_general(results.best_estimator_, np.hstack([stored_psych_model[0]['sig_input'], stored_psych_model[0]['sig_input']]), test_model.encoder.transform(stored_psych_model[1]), scorer=brier_score_loss, learning_type='lpu')

# %%
# test_pipeline['clf'].predict_proba(stored_psych_model[0]) - stored_psych_model[1]#, stored_psych_model[0]['sig_input']]), test_model.encoder.transform(stored_psych_model[1]), scorer=brier_score_loss, learning_type='real')
current_l = (stored_psych_model[1] / 2).astype(int)

temp = np.abs(test_pipeline['clf'].predict_proba(stored_psych_model[0]) - current_l.reshape((-1, 1)))
temp[temp <1e-200] = 0
np.mean(-temp ** 2)

# %%
for key, value in results.cv_results_.items():
    if 'score' in key and '_brier' in key and 'rank' not in key:
        big_values_idx = value > -.1#         print (key, value)
        top_idx = value.argsort()[-5:][::-1]+1
        print (key, np.arange(len(big_values_idx))[top_idx-1], value[top_idx-1], top_idx)

# %%
import pandas as pd
# pd.options.display.max_columns = 1

sample_dict = {'a': [1, 2, 3], 'b':[3,4, 5]}
df = pd.DataFrame(sample_dict)

# %%
import qgrid
qgrid_widget = qgrid.show_grid(df, show_toolbar=True)
qgrid_widget

# %%


# %%
results.best_index_

# %%
for key, value in results.cv_results_.items():
    if 'score' in key and '_f' in key and 'rank' not in key:
        big_values_idx = value >-0.2
#         print (key, value)
        print (key, np.arange(len(big_values_idx))[big_values_idx], value[big_values_idx])
#     prin

# %%
results.best_index_

# %%
test_pipeline['clf'] == test_model

# %%
# ax.hist(kernel_mat[:, 0], bins=50)
# import numpy as np
# from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
# from gpflow.kernels import SquaredExponential
# sq_kernel = SquaredExponential(lengthscales=23.10129700083158, variance=0.7054802310718645)
# # from sklearn.metrics.pairwise import r
# from math_utils import modified_rbf_kernel, modified_linear_kernel
# a = np.asarray([[1., 1.256], [1, 2.], [3., 3], [40., 33]])
# # b =  np.asarray([[1., .25], [1, 13]])
# # print(rbf_kernel(a,a, gamma=10.) * 0.2)
# print(results.best_estimator_['clf'].gp_kernel_amplitude)
# print(modified_linear_kernel(a,a, var=results.best_estimator_['clf'].gp_kernel_amplitude))
# print(sq_kernel(a, a))

# %%
# print (results.best_estimator_['clf'].set_params(**{'sig_a_init':0.2}))
# print (results.best_estimator_['clf'].get_params())
fig, ax = plt.subplots(1)
ax.plot(results.best_estimator_['clf'].parameter_reporter.alpha_norm_1_hist, label=r'$||\alpha||_1$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.alpha_norm_inf_hist, label=r'$||\alpha||_\inf$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.alpha_norm_neg_inf_hist, label=r'$||\alpha||_{-\inf}$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.alpha_norm_2_hist, label=r'$||\alpha||_2$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.a_norm_1_hist, label=r'$||a||_1$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.a_norm_2_hist, label=r'$||a||_2$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.a_norm_inf_hist, label=r'$||a||_\inf$')
ax.plot(results.best_estimator_['clf'].parameter_reporter.alpha_norm_neg_inf_hist, label=r'$||a||_{-\inf}$')
ax.set_xlabel('Epochs')
fig.legend()
# print(results.best_estimator_['clf'].parameter_reporter.a_norm_1_hist)

# %%


# %%
fig, ax = plt.subplots(2)
ax[0].plot(results.best_estimator_['clf'].history['lpu_f1_score_for_l'],label='l beta score')
ax[0].plot(results.best_estimator_['clf'].history['lpu_f1_score_for_y'],label='y beta score')
ax[0].plot(results.best_estimator_['clf'].parameter_reporter.gamma_hist, label='gamma')
ax[0].plot(results.best_estimator_['clf'].parameter_reporter.lambda_hist, label='lambda')# ax.show()
# ax.set_ylim(-10, 100.)
ax[1].plot(np.asarray(results.best_estimator_['clf'].history['loss']), label='loss')
ax[1].set_yscale('log')
fig.legend()
# plt.xlim(0, 10.)


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
np.exp(-12845.056)

# %%


# %%


# %%


# %%


# %%
# %history -g -f fake_reviews_2.ipynb

# %%
from scipy.special import expit

# %%
1. - expit(-30)

# %%
# from __future__ import print_function
# %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats.distributions
# from msmbuilder.example_datasets import load_quadwell
# from msmbuilder.example_datasets import quadwell_eigs
# from msmbuilder.cluster import NDGrid
# from msmbuilder.msm import MarkovStateModel
# from sklearn.pipeline import Pipeline
# from sklearn.grid_search import RandomizedSearchCV

# %%
print(__doc__)

import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from time import time
from scipy.stats import randint as sp_randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target
X = X[y<2]
y = y[y<2]
print (y)
# build a classifier
clf = RandomForestClassifier(n_estimators=20)
from sklearn.preprocessing import StandardScaler
pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', clf)])


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_roc_auc'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_roc_auc'][candidate],
                  results['std_test_roc_auc'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"clf__max_depth": [3, None],
              "clf__max_features": sp_randint(1, 11),
              "clf__min_samples_split": sp_randint(2, 11),
              "clf__min_samples_leaf": sp_randint(1, 11),
              "clf__bootstrap": [True, False],
              "clf__criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 2
random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                   n_iter=n_iter_search, refit='roc_auc', scoring={'roc_auc':make_scorer(roc_auc_score), 'aps':make_scorer(average_precision_score)})

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


# %%
random_search.fit(X, y)


# %%
random_search.fit(X, y)


# %%
a

# %%
np.flatnonzero(a == 3)[0]

# %%
random_search.cv_results_['rank_test_roc_auc']

# %%
a = [] + list(np.flatnonzero(random_search.cv_results_['rank_test_roc_auc'] == 3))

# %%
a

# %%
b = a[:2]

# %%
b

# %%
random_search.best_estimator_.named_steps['clf']

# %%
import dask.bag as db

bag = db.from_sequence(range(6))

# %%


# %%
# Create a function for mapping
from dask.distributed import Client
def f(x):
    return x**2
client = Client(address_and_port)#, serializers=['pickle'],deserializers=['pickle'])
client.restart()
client.wait_for_workers()
# Create the map and compute it
results = bag.map(f).compute()

# %%
results

# %%
list_of_dicts = [{'b':12 , 'c':13}, {'b': 35, 'c':15}, {'b':-100, 'c': 33}, {'b':35, 'c':100}]

# %%
max(list_of_dicts, key=lambda x: x['b'])

# %%
list(zip([1, 2, 3], [3, 4, 8], [9, 8, 12]))

# %%
layer = tf.keras.layers.Softmax()

# %%
layer(np.asarray([0, .2, .2]))

# %%
import tensorflow_probability as tfp
import tensorflow as tf
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
    amplitude=None, length_scale=None, feature_ndims=1, validate_args=False,
    name='ExponentiatedQuadratic'
)

# %%
a = tf.convert_to_tensor([[1, 2, 3], [3, 4, 5]], dtype=tf.float32)
b = tf.convert_to_tensor([[1, 2, 3], [4.5,  1, 1]], dtype=tf.float32)

kernel.matrix(a, b)

# %%


# %%
from klepto.archives import *
arch = file_archive('foo.txt')
arch['kernel'] = kernel

# %%
# look at the "on-disk" copy
arch

# %%
import tensorflow as tf
import numpy as np
a = tf.convert_to_tensor(np.arange(9).reshape((3, 3)))

# %%
b = tf.reshape(a, (-1, 1))

# %%
b + [1]

# %%
tf.reduce_sum(a, axis=1)

# %%
tf.expand_dims([1], axis=0)

# %%
tf.shape(tf.tile([[2]], [2, 3]))

# %%
from sklearn.preprocessing import StandardScaler
def check_change(a):
    clf = StandardScaler()
    a = list(a) + [3]
    print(a)

# %%
b = np.asarray([[1, 2]])

# %%
check_change(b)

# %%
b

# %%


# %%



