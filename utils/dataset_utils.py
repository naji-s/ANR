from glob import glob
from os import path as os_path
from os import makedirs, walk
from os import path as os_path
from numpy import genfromtxt, asarray, zeros
from pandas import Series as pd_Series
from pandas import DataFrame as pd_df
from pandas import read_pickle as pd_read_pickle
from dill import load as pickle_load
from dill import dump as pickle_dump
from scipy.special import expit
import sys
# sys.path.append('/home/scratch/nshajari/psych_model')
# sys.path.append('/home/scratch/nshajari/psych_model/datasets')
sys.path.append('/home/scratch/nshajari/psych_model/datasets/animal_no_animal')
# sys.path.append('/home/scratch/nshajari/psych_model/datasets/animal_no_animal/utils')

from animal_no_animal_utils import extract_features, subject_related_data_read
from sklearn.datasets import make_spd_matrix
import numpy as np
def read_fake_reviews(loc=None, subject=1, mode='cluster', process_id='TEST', embedding=None):
    import dill as pickle
    # subject number
    if subject != 'all':
        i = subject_num = int(subject)
    else:
        i = subject_num = subject
        
    if mode == 'client':
        pickle_file_location = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/fake_reviews/fake_reviews_tf_idf.pkl'
        output = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/fake_reviews/output/'
    elif mode == 'cluster':
        pickle_file_location = '/home/nshajari/psych_model/datasets/fake_reviews/fake_reviews_tf_idf.pkl'
        output = '/home/nshajari/psych_model/datasets/fake_reviews/output/'
    elif mode == 'auton':
        pickle_file_location = '/home/scratch/nshajari/psych_model/datasets/fake_reviews/fake_reviews_tf_idf.pkl'
        output = '/home/scratch/nshajari/psych_model/datasets/fake_reviews/output/'
    else:
        raise NotImplementedError

    if not os_path.exists(output + str(subject_num) + '/' + process_id):
           makedirs(output + str(subject_num) + '/' + process_id)

    #     else: 
    #         random_state =  random.get_state()




    #     import os
    #     os.environ['KMP_AFFINITY'] = 'proclist=[0, 64, 128, 192],explicit'
    #     os.environ['OPENBLAS_NUM_THREADS'] = '1'
        # def main(profile, subject, mode, backend='dask', extract_features=False, pickle_file_location = 'tf_idf.pkl'):
    """
    :param output_dir: String, the location of where the final information will be written
    :param subject_name: String, the encoded name of subject to extract data
    :return:
    """

    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    extract_features=True
    pickle_file_location = 'tf_idf.pkl'

    import os
    import dill as pickle
    if extract_features or not(os_path.isfile(pickle_file_location)):
        print ("WE ARE EXTRACTING FEATURES")
        my_data = genfromtxt(output[:-7] + 'op_spam.human_judgements.tsv', delimiter='\t', dtype=None, encoding='utf-8')
        label_dict = {'T': False, 'F': True}
        data_dict = dict()
        data_dict['y'] =  asarray([label_dict[item] for item in my_data[:, 0]]).astype(int)
        data_dict['hotel_name'] =  asarray([str.strip(str(item)) for item in my_data[:, 1]])
        data_dict['s_1'] =  asarray([label_dict[item] for item in my_data[:, 2]]).astype(int)
        data_dict['s_2'] =  asarray([label_dict[item] for item in my_data[:, 3]]).astype(int)
        data_dict['s_3'] =  asarray([label_dict[item] for item in my_data[:, 4]]).astype(int)

        # tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        data_dict['X_text'] = [str(item).lower() for item in my_data[:, 5]]

        ## write
    #         bytes_out = pickle.dumps(data_dict)
    #         with open(pickle_file_location, 'wb') as f_out:
    #             for idx in range(0, n_bytes, max_bytes):
    #                 f_out.write(bytes_out[idx:idx + max_bytes])
    else:
        ## read
        bytes_in = bytearray(0)
        input_size = os_path.getsize(pickle_file_location)
        with open(pickle_file_location, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        data_dict = pickle.loads(bytes_in)

    if subject == 'all':
        l_dict = dict()
        r_dict = dict()
        for i in range(3):
            l = asarray(data_dict['s_'+str(i+1)]) & asarray(data_dict['y'])
            r = asarray(data_dict['s_'+str(i+1)]) 
            r_dict[i] = r
            l_dict[i] = l
        
        new_l = zeros(len(l)).astype(bool)
        new_r = zeros(len(r)).astype(bool)
        for i in range(3):
            new_l = new_l | l_dict[i].astype(bool)
            new_r = new_r | r_dict[i].astype(bool)
        main_l = pd_Series(new_l.astype(int))
        main_r = pd_Series(new_r.astype(int))
    else:
        main_l = pd_Series(asarray(data_dict['s_'+str(subject)]) & asarray(data_dict['y']))
        main_r = pd_Series(asarray(data_dict['s_'+str(subject)]))
    if embedding == 'BERT':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        main_X = model.encode(pd_Series(data_dict['X_text']))
    else:
        main_X = pd_Series(asarray(data_dict['X_text']))
    main_y = pd_Series(asarray(data_dict['y']))
#     main_l = pd_Series(asarray(data_dict['l']))
    
    return main_X, main_y, main_r, main_l


def read_fake_reviews_2(loc=None, subject=1, mode='cluster', process_id='TEST', embedding=None):
    import dill as pickle
    # subject number
    if subject != 'all':
        i = subject_num = int(subject)
    else:
        i = subject_num = subject
        
    if mode == 'client':
        pickle_file_location = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/fake_reviews_2/fake_reviews_2_tf_idf.pkl'
        output = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/fake_reviews_2/output/'
    elif mode == 'cluster':
        pickle_file_location = '/home/nshajari/psych_model/datasets/fake_reviews_2/fake_reviews_2_tf_idf.pkl'
        output = '/home/nshajari/psych_model/datasets/fake_reviews_2/output/'
    elif mode == 'auton':
        pickle_file_location = '/home/scratch/nshajari/psych_model/datasets/fake_reviews_2/fake_reviews_2_tf_idf.pkl'
        output = '/home/scratch/nshajari/psych_model/datasets/fake_reviews_2/output/'
    else:
        raise NotImplementedError

    if not os_path.exists(output + str(subject_num) + '/' + process_id):
           makedirs(output + str(subject_num) + '/' + process_id)

    #     else: 
    #         random_state =  random.get_state()




    #     import os
    #     os.environ['KMP_AFFINITY'] = 'proclist=[0, 64, 128, 192],explicit'
    #     os.environ['OPENBLAS_NUM_THREADS'] = '1'
        # def main(profile, subject, mode, backend='dask', extract_features=False, pickle_file_location = 'tf_idf.pkl'):
    """
    :param output_dir: String, the location of where the final information will be written
    :param subject_name: String, the encoded name of subject to extract data
    :return:
    """

    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    extract_features=True
    pickle_file_location = 'tf_idf.pkl'

    import os
    import dill as pickle
    if extract_features or not(os_path.isfile(pickle_file_location)):
        print ("WE ARE EXTRACTING FEATURES")
        my_data = genfromtxt(output[:-7] + 'Negative_Opinion_Spam.tsv', delimiter='\t', dtype=None, encoding='utf-8', comments=None, skip_header=True)
        label_dict = {'T': False, 'F': True}
        data_dict = dict()
        data_dict['y'] =  asarray([label_dict[item] for item in my_data[:, 0]]).astype(int)
        data_dict['hotel_name'] =  asarray([str.strip(str(item)) for item in my_data[:, 1]])
        data_dict['r_1'] =  asarray([label_dict[item] for item in my_data[:, 3]]).astype(int)
        data_dict['r_2'] =  asarray([label_dict[item] for item in my_data[:, 4]]).astype(int)
        data_dict['r_3'] =  asarray([label_dict[item] for item in my_data[:, 5]]).astype(int)

        # tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        data_dict['X_text'] = [str(item).lower() for item in my_data[:, 2]]

        ## write
    #         bytes_out = pickle.dumps(data_dict)
    #         with open(pickle_file_location, 'wb') as f_out:
    #             for idx in range(0, n_bytes, max_bytes):
    #                 f_out.write(bytes_out[idx:idx + max_bytes])
    else:
        ## read
        bytes_in = bytearray(0)
        input_size = os_path.getsize(pickle_file_location)
        with open(pickle_file_location, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        data_dict = pickle.loads(bytes_in)

    if subject == 'all':
        l_dict = dict()
        r_dict = dict()
        for i in range(3):
            l = asarray(data_dict['r_'+str(i+1)]) & asarray(data_dict['y'])
            r = asarray(data_dict['r_'+str(i+1)]) 
            r_dict[i] = r
            l_dict[i] = l
        
        new_l = zeros(len(l)).astype(bool)
        new_r = zeros(len(r)).astype(bool)
        for i in range(3):
            new_l = new_l | l_dict[i].astype(bool)
            new_r = new_r | r_dict[i].astype(bool)
        main_l = pd_Series(new_l.astype(int))
        main_r = pd_Series(new_r.astype(int))
    else:
        main_l = pd_Series(asarray(data_dict['r_'+str(subject)]) & asarray(data_dict['y']))
        main_r = pd_Series(asarray(data_dict['r_'+str(subject)]))
    if embedding == 'BERT':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        main_X = model.encode(pd_Series(data_dict['X_text']))
    else:
        main_X = pd_Series(asarray(data_dict['X_text']))
    main_y = pd_Series(asarray(data_dict['y']))
#     main_l = pd_Series(asarray(data_dict['l']))
    
    return main_X, main_y, main_r, main_l

def read_swissprot(loc=None, mode='cluster', process_id='TEST', embedding=None):
    import dill as pickle
    def text_extractor(path):
        # token_dict = OrderedDict()
        # keys = OrderedDict()
        text_list = []
        target_list = []
        counter = 0
        for subdir, dirs, files in walk(path):
            direct = subdir.split('/')[-1]

            if direct == '':
                continue
            # keys[direct] = OrderedDict()
            for file in files:
                if file=='.DS_Store':
                    continue
                if 'checkpoint' in file:
                    continue
                file_path = subdir + os_path.sep + file
                shakes = open(file_path, 'r')
                text = shakes.read()
#                 # lowercasing and taking out the punctuation
#                 lowers = text.lower()
                # storing the no_token version of each text file (data point or X)
                text_list.append(text)

                # storing the index for the file based on directory (therefore U, P, Q) and then based
                # on file which is data name
                # ke/ys[direct][file] = counter

                target_list.append(direct)
                # print (direct, file)
                counter += 1
        return target_list, text_list
    
    
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    pickle_file_location = '/home/scratch/nshajari/psych_model/datasets/swissprot.data/'
    extract_features = False
    if extract_features or not(os_path.isfile(pickle_file_location + 'protein_tfidf.pkl')):
        targets, X_text = text_extractor(pickle_file_location)
#         print ("salam")
        ## write
        bytes_out = pickle.dumps([X_text, targets])
        with open(pickle_file_location + 'protein_tfidf.pkl', 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])
    else:
#         print ("salam 2")
        ## read
        bytes_in = bytearray(0)
        input_size = os_path.getsize(pickle_file_location + 'protein_tfidf.pkl')
        with open(pickle_file_location + 'protein_tfidf.pkl', 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        X_text, targets = pickle.loads(bytes_in)
#     print (X_text[:5], targets[:5])
    LePU_label_dict= {'P': 1, 'Q': 0, 'N': 0}
    real_label_dict= {'P': 1, 'Q': 1, 'N': 0}
    y = []
    l = []
    for target in targets:
        if target in ['P', 'Q', 'N']:
            y.append(real_label_dict[target])
            l.append(LePU_label_dict[target])
    y = np.asarray(y)
    l = np.asarray(l)

    if embedding == 'BERT':
        if not(os_path.isfile(pickle_file_location + 'protein_tfidf_BERT.pkl')):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            main_X = model.encode(pd_Series(X_text))
        else:
            with open(pickle_file_location + 'protein_tfidf_BERT.pkl', 'rb') as f:
                main_X, main_y, main_l = pickle.load(f)
    else:
        main_X = asarray(X_text)
        main_y = pd_Series(asarray(y))
        main_l = pd_Series(asarray(l))
    
    

    return main_X, main_y, main_l
def read_neuroscience(loc=None, subject=None, mode=None, process_id=None, model='vgg', reduce_dim=False, extracting_layer='fc1', reverse=False):
    import os
    import pickle
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    """
    :param output_dir: String, the location of where the final information will be written
    :param subject: String, the encoded name of subject to extract data
    :return:
    """
    if subject is None:
        subject = 'hth'
    if mode is None:
        mode = 'cluster'
    if process_id is None:
        process_id = 'TEST'
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
#     subject = 'hth'
    backend='dask'
    is_extract_features=False
    pickle_file_location = 'tf_idf.pkl'
    
    
    if mode == 'client':
        pickle_file_location = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/fake_reviews/fake_reviews_tf_idf.pkl'
        output = '/Users/naji/Box Sync/Box Sync/CMU/Masters Thesis/psych_model/datasets/animal_no_animal/output/'
    elif mode == 'cluster':
        pickle_file_location = '/home/nshajari/psych_model/datasets/animal_no_animal/animal_no_animal_tf_idf.pkl'
        output = '/home/nshajari/psych_model/datasets/animal_no_animal/output/'
    elif mode =='auton':
        pickle_file_location = '/zfsauton2/home/nshajari/psych_model/datasets/animal_no_animal/animal_no_animal_tf_idf.pkl'
        output = '/zfsauton2/home/nshajari/psych_model/datasets/animal_no_animal/output/'        
    else:
        raise NotImplementedError

    output_list = glob(output + "*.txt")
    output_file_name = 'trial ' + str(len(output_list)) + '.txt'
#     print("This is trail " + output_file_name)
    output_location = output + output_file_name

    ########
#     print ("checking to see if the dump exist:", output[:-7]+'subject_data/vgg_16_extract_dumps/'+subject+'_' + extracting_layer + '_reverse_' +str(reverse)+'.pkl')
    print ("HERE", output[:-7]+'subject_data/vgg_16_extract_dumps/'+subject+'_' + extracting_layer + '_reverse_' +str(reverse)+'_'+model+'.pkl')

    if is_extract_features or not(os_path.isfile(output[:-7]+'subject_data/vgg_16_extract_dumps/'+str(subject)+'_' + extracting_layer + '_reverse_' +str(reverse)+'_'+model+'.pkl')):
        subject_related_X_2d_RGB_arr, Y_or_real_label_arr, subject_related_response_arr, S_or_subject_related_pos_label_arr = \
            extract_features(output_location=output[:-7].replace('/zfsauton2/home', '/home/scratch')+'abimal_no_animal/subject_data/vgg_16_extract_dumps/', reverse=reverse, subject=subject,extracting_layer=extracting_layer, model=model)
        data = asarray(
            [subject_related_X_2d_RGB_arr, Y_or_real_label_arr[None, ...], subject_related_response_arr[None, ...],
             S_or_subject_related_pos_label_arr[None, ...]])
        file_name = output[:-7] + 'subject_data/vgg_16_extract_dumps/' + subject + '_' + extracting_layer + '_reverse_' + str(reverse) + '_' + model + '.pkl'
        if not os_path.isfile(file_name):
            with open(file_name,'wb') as file:
                pickle.dump(data, file)
        else:
            random_suffix = np.random.randn(1)
            print ("FILE EXIST... creating a new file with suffix", random_suffix)
            with open(file_name[:-4] + str(random_suffix)+filename[-4:],'wb') as file:
                pickle.dump(data, file)
        X, y, r, l = subject_related_X_2d_RGB_arr, Y_or_real_label_arr, subject_related_response_arr, S_or_subject_related_pos_label_arr
    else:
        if reduce_dim:
            with open(output[:-7]+'subject_data/vgg_16_extract_dumps_dim_reduced/'+subject+'_' + extracting_layer + '_reverse_' +str(reverse)+'_'+model+'_reduced_dim.pkl', 'rb') as f:
                X, y, r, l, X_dim_reduced = pickle.load(f)
        else:
            with open(output[:-7]+'subject_data/vgg_16_extract_dumps/'+subject+'_' + extracting_layer + '_reverse_' +str(reverse)+'_'+model+'.pkl', "rb") as f:
                X, y, r, l = pickle.load(f)
            

#             subject_related_X_2d_RGB_arr, Y_or_real_label_arr, subject_related_response_arr, S_or_subject_related_pos_label_arr = pickle.load(open(output[:-7]+'subject_data/vgg_16_extract_dumps/'+subject_name+'_' + extracting_layer + '_reverse_' +str(reverse)+'.pkl', "rb"))

#         X, y, r, l = pd_read_pickle(output[:-7]+'subject_data/vgg_16_extract_dumps/'+subject+'_' + extracting_layer + '_reverse_' +str(reverse)+'.pkl')
#     print (X.squeeze())
    main_X = X.squeeze().astype(np.float64)
    main_y = pd_Series(asarray(y).squeeze())
    main_l = pd_Series(asarray(l).squeeze())
    main_r = pd_Series(asarray(r).squeeze())
    if reduce_dim:
        return main_X, main_y, main_r, main_l, X_dim_reduced.astype(np.float64)
    else:
        return main_X, main_y, main_r, main_l
    
# if __name__ == '__main__':
#     main_X, main_y, _, main_l = read_neuroscience(subject='hth')
#     print (main_X[:5], main_y[:20], main_l[:20])
#     read_swissprot()

from sklearn.datasets import make_blobs

# global w_1, w_2, Sigma_1, Sigma_2, b_1, b_2, real_psych_gamma, real_psych_lambda, params_initialized
# params_initialized = False

def create_synthetic_dataset(X=None, y=None, sample_size=1000, dim=5, store_params=None, initial_lambda_range=None, initial_gamma_range=None, zero_alpha=False, zero_beta=False):
    X_is_not_None = X is not None
    y_is_not_None = y is not None
    print ("HELLO!")
    if not(X_is_not_None and y_is_not_None) and (X_is_not_None or y_is_not_None):
        raise NotImplementedError("X and y are supposed to be set together or neither should be set")
    if X_is_not_None and y_is_not_None:
#         print ("sample_size is setting to the length of y argument, y.shape[0]...")
        sample_size = y.shape[0]
#         print ("dim is setting to the dim of X argument, X.shape[-1]...")
        dim = X.shape[-1]
    if store_params is None:
        store_params = False   
    else:
        store_params = store_params
#     if store_params:
#         global w_1, w_2, Sigma_1, Sigma_2, b_1, b_2, real_psych_gamma, real_psych_lambda, params_initialized
    
#     X = make_blobs(n_samples=sample_size, n_features=dim, centers=100, cluster_std=.01, center_box=(-100.0, 100.0), shuffle=True, random_state=None)[0]
#     print ("dim is: ", dim)
#        X = np.random.uniform(low= -10, high=10, size=sample_size * dim).reshape((sample_size, dim))
    params_initialized = False
    if not params_initialized or not store_params:
        if not(X_is_not_None and y_is_not_None):
            X = (np.random.randn(sample_size * dim)).reshape((-1, dim)).astype(np.float64) * 5
    #        print ("Head of X:", X[:5, :])
        Sigma_1 = make_spd_matrix(dim, random_state=None)
        Sigma_1 = Sigma_1 + np.diag(abs(np.random.binomial(n=1, p=0.5, size=dim))) #* np.max(np.diag(Sigma_1))
        Sigma_2 = make_spd_matrix(dim, random_state=None)
        Sigma_2 = Sigma_2 + np.diag(abs(np.random.binomial(n=1, p=0.5, size=dim))) #* np.max(np.diag(Sigma_2)) * 3
        w_1 = 5 * np.random.randn(dim).astype(np.float64) # + (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) 
    #        w_1 = (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * 10
        #         w_1 = w_1.T.dot(Sigma_1) + (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) 
    #        w_1 = (np.random.binomial(n=1, p=0.5, size=dim)) * 0.2
    #        w_2 = (np.random.binomial(n=1, p=0.5, size=dim)) * .5
    #        w_1 = w_1.astype(np.float64) 
    #        w_2 = (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * 10
    #        w_1 = w_1.T.dot(Sigma_1) #+ (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * .5
    #        w_2 = w_2.T.dot(Sigma_2) #+ (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * .5
    #        w_2 = (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * 5# + w_2.T.dot(Sigma_2) + 
        b_1 = np.float64(np.random.randn())
        w_2 = 5 * np.random.randn(dim).astype(np.float64)# +  20 * (np.random.binomial(n=1, p=0.5, size=dim)-0.5)
        b_2 = np.float64(np.random.randn()) * 5#+  10 * (np.random.binomial(n=1, p=0.5, size=1)-0.5)).squeeze()
        initial_gamma = np.random.uniform(*initial_gamma_range)
        initial_lambda = np.random.uniform(*initial_lambda_range)
        counter = 0
        response_sig = expit(w_1.dot(X.T) + b_1)
        y = np.random.binomial(p=response_sig, n=1)
        while np.linalg.norm(w_2) < 5 or (abs(b_2) / np.abs(w_2) > 1.).any():# or (response < 0.5).mean() < 0.3 or (response > 0.5).mean() < 0.3:
            if counter and np.mod(counter, 1000) == 0:
                print ("fuckeeeee", counter, np.std(X))
            w_2 = 5 * np.random.randn(dim).astype(np.float64)# +  20 * (np.random.binomial(n=1, p=0.5, size=dim)-0.5)
            b_2 = np.float64(np.random.randn()) * 5  #+  10 * (np.random.binomial(n=1, p=0.5, size=1)-0.5)).squeeze()
            response = expit(w_2.dot(X[y==1].T) + b_2)
            counter += 1
            
    #        while np.dot(w_1, w_2) < 0 or ((np.dot(w_1, w_2) / (np.linalg.norm(w_1) * np.linalg.norm(w_2))) < 0.5):
    #            w_1 = np.random.randn(dim).astype(np.float64) * 0.1
    # #            w_1 = w_1.T.dot(Sigma_1) + (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * 2
    #            # w_1 = w_1.astype(np.float64) 
    #            w_2 = np.random.randn(dim).astype(np.float64) * 0.3
    #            w_2 = w_2.T.dot(Sigma_2) + (np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1) * 1
    #        w_1 = np.asarray([2.4, 2.4, 2.2, 3.4,    4.0, 1.8, 1.9 , 1.8 , 1.0, 1.2]) * 2.0
    #        w_2 = np.asarray([-5., 0., -0.10, 0.1, -.5,   -7.3,     .05,  -0.1, -0.1, 0.]) * 2.0
    #        w_1 = np.asarray([0.1, 0])#, .10])
    #        w_2 = np.asarray([0., .2])#, 0])
        # w_2 = w_1 + np.random.randn(dim) * 10
        # w_1 = w_2.astype(np.float64)
    #        print ("Original w: ", w_1, w_2)
    #        global func_3_initialized
    #        global func_4_initialized
    #        func_3_initialized = False
    #        func_4_initialized = False
    #        global kernel_centers
    #        global kernel_widths
    #        global func_5_initialized
    #        func_5_initialized = False
    #        global w_list
    #        global b_list
    #        w_list = []
    #        b_list = []
    #        def rand_func_3(x, kernel_num=3, t_dom_start=-5, t_dom_end=5):
    #            global func_3_initialized
    #            global kernel_centers
    #            global kernel_weights
    #            if np.ndim(x) == 1:
    #                x = x[None, ...]
    #            dim = x.shape[-1]
    #            if not func_3_initialized:
    #                # .2 is to shrink the kernel centers to fall completely in the interval
    #                kernel_centers = np.random.randn(kernel_num * dim).reshape((kernel_num, dim)) * 5
    #                kernel_weights = np.random.randn(kernel_num * dim).reshape((-1, dim)).squeeze() * 10
    # #                kernel_centers = np.random.uniform(t_dom_start, t_dom_end, size=kernel_num)
    # #                kernel_weights = np.random.uniform(0., 15, size=kernel_num * dim).reshape((-1, dim)).squeeze()
    #                func_3_initialized = True
    #            output = np.zeros(x.shape[0])
    #            for c, w in zip(kernel_centers, kernel_weights):
    #                if np.ndim(w) == 0:
    #                    w = np.asarray([w])
    #                output += expit(np.dot(x - c, w))
    #            return (output / len(kernel_weights)).squeeze()
    #        def characteristic_func(x=None, w=None, c=None):
    #            # print ("x:", x, "w:", w, "c:", c)
    #            # print ((x > c - w / 2.))
    #            # print ("and all is:", (x > c - w / 2).all() )
    #            if (x > c - w / 2.).all() and (x < c + w / 2.).all():
    #                return 1
    #            else:
    #                return 0
    #        def rand_func_4(x, kernel_num=10):
    #            global func_4_initialized
    #            global kernel_centers
    #            global kernel_widths
    #            dim = x.shape[-1]
    #            if not func_4_initialized:
    #                # .2 is to shrink the kernel centers to fall completely in the interval
    #                kernel_centers = np.random.uniform(t_dom_start, t_dom_end , size=kernel_num)
    #                kernel_widths = np.random.uniform(.1, 10., size=kernel_num * dim).reshape((-1, dim)).squeeze()
    #                func_4_initialized = True
    #            output = 0
    #            # output = np.zeros(x.shape[0])
    #            for c, w in zip(kernel_centers, kernel_widths):
    #                if np.ndim(w) == 0:
    #                    w = np.asarray([w])
    #                output += characteristic_func(x, w, c)
    #            if output > 1:
    #                return 1
    #            return output
    #        def rand_func_5(x, kernel_num=10):
    #            global func_5_initialized
    #            global w_list
    #            global b_list
    #            temp = np.zeros(x.shape[0])
    #            if not func_5_initialized:
    #                for i in range(kernel_num):
    #                    Sigma_temp = make_spd_matrix(dim, random_state=None) 
    #                    w_temp = np.random.randn(dim).astype(np.float64) * (i + 1) 
    #                    w_list.append(abs(w_temp))
    #                    b_list.append(np.random.binomial(n=1, p=0.5))
    # #                    w_temp = w_temp.T.dot(Sigma_temp) 
    #            for w, b in zip(w_list, b_list):
    #                temp = temp + expit(w.dot(x.T) + b)
    #            temp = temp / kernel_num
    #            return temp
        # p =  expit(np.dot(w_1, X.T)).astype(np.float64)
        # y_1_X_idx = np.random.binomial(1., p=p).astype(bool)
    #            return rand_func_5(x)
    #            return expit(w_1.dot(x.T) + b_1)
    #        y = np.random.binomial(1., p=t(X)).astype(bool)
    #        y_1_X_idx = np.arange(len(y_1_X_idx))[y_1_X_idx]
    #        print ("OHHHH:", X.shape)
        # y_1_X_idx = np.random.binomial(1., p=expit(np.dot(w_1, X.T)).astype(np.float64)).astype(bool)
    #        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_set_ratio, random_state=42)
    #        print ("y trains:", y_train)
    #        print ("X train and test shapes are:", X_train.shape, X_test.shape)
    #        y_1_X_train = X_train[y_1_X_idx_train]
    #        y_1_X_test = X_test[y_1_X_idx_test]
    #        y_test = np.zeros(X_test.shape[0])
    #        y_test[y_1_X_idx_test] = 1.
    #        y_train = np.zeros(X_train.shape[0])
    #        y_train[y_1_X_idx_train] = 1.
    #        y_train = pd.DataFrame(data=y_train)
    #        real_psych_gamma = np.random.uniform(.1, .2)
    #        real_psych_lambda = np.random.uniform(.1, .2)
    #        real_psych_gamma = np.random.uniform(0.1, .6)
    #        real_psych_lambda = np.random.uniform(0.01, .2)
        if initial_gamma is None:
            real_psych_gamma = 0.15
        else:
            real_psych_gamma = initial_gamma
            
        if initial_lambda is None:
            real_psych_lambda = 0.05
        else:
            real_psych_lambda = initial_lambda
    #        while (real_psych_gamma + real_psych_lambda >= .99) or real_psych_lambda > 0.1:
    #            real_psych_gamma = np.random.uniform(.01, .99)
    #            real_psych_lambda = np.random.uniform(.01, .99)
    #        real_psych_gamma = 0.5
    #        real_psych_lambda = 0.05
    #        real_psych_gamma = 0.05
    #        real_psych_lambda = 0.15
    #        real_psych_lambda
#         params_initialized = True
    epsilon = .0
#     w_1 = np.asarray([  2.71570128, -25.93288602,  -6.52176385,  14.81049625,
#      2.91476569,  -4.85671258,  12.76502629,  14.27152726,
#      1.1281678 ,  12.0360221 ])
#     b_1 = 0.7137360484493082
#     w_2 = np.asarray([  5.21371575,  12.40951289,  13.89091276,   7.69818577,
#        20.59733184,  -8.02277932,  13.85381487,  -9.86034087,
#     12.58786148,   5.18390823])
#     b_2 = 0.5811576889647186
    def t(x):
        if np.ndim(x) == 1:
            x = x.reshape((-1, 1))
        return (1-epsilon) * expit(w_1.dot(X.T) + b_1)# + epsilon * rand_func_5(X) + 
    if (X_is_not_None and y_is_not_None):
        w_1 = None
        b_1 = None
    else:
        y = np.random.binomial(1., p=t(X)).astype(bool)
    if zero_alpha:
        w_2 = np.zeros_like(w_2)
    if zero_beta:
        b_2 = np.zeros_like(b_2)
    def psychometric_func(x):
        if np.ndim(x) == 1:
            x = x.reshape((-1, 1))
        #       return expit(np.dot(w_2, x.T) + b_2)
        #       print ("x has shape", x.shape)
        #       print ("w_2 has shape", w_2)
        return real_psych_gamma + (1. - real_psych_gamma - real_psych_lambda) * expit(w_2.dot(x.T) + b_2)
    y = y.astype(bool)
    l = np.zeros(y.shape)
#     print("zfzfff:", y.shape, l.shape)
    l[y] = np.random.binomial(1., p=psychometric_func(X[y]).astype(np.float64)).astype(bool)
    l[~y] = 0
    l = l.astype(bool)
#     noise_2 = np.random.binomial(1., p=[0.1]*len(y)).astype(bool)
#     l[noise_2] = ~l[noise_2]
#     l = l.reshape((1, -1)).squeeze()
    params_initialized
    return X, pd_Series(y), pd_Series(l), [real_psych_gamma, real_psych_lambda, w_1, b_1, w_2, b_2]