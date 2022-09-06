from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


def stem_tokens(tokens, stemmer):
    """ Stem a set of tokens and return the result"""
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# def tokenize(text):
#     """convert a given text to an array of tokens"""
#     stemmer = PorterStemmer()
#     # setting a regex to remove punctuation
#     # regex_punc_remover_tokenizer = RegexpTokenizer(r'(\w+)\.{3,}')
#     pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
#     exclude = set(string.punctuation)
#     # table = str.maketrans("", "")
#     # removed_punctuation = text.translate(table)
#     # removed_space = re.sub(' +', ' ', removed_punctuation)
#     # removed_space = removed_punctuation
#     # regex_punc_remover_tokenizer = RegexpTokenizer(r'\w+')
#     # tokens = regex_punc_remover_tokenizer.tokenize(removed_space)
#     # tokens
#     stems = stem_tokens(pattern, stemmer)
#     return stems

import re
REGEX = re.compile(r",\s*")
from nltk.tokenize import word_tokenize
# def tokenize(text):
    # return word_tokenize(text)
    # return None
    # return [tok.strip().lower() for tok in REGEX.split(text)]
tokenize = None
def text_extractor(path):
    # token_dict = OrderedDict()
    # keys = OrderedDict()
    text_list = []
    idx_list = []
    counter = 0
    for subdir, dirs, files in os.walk(path):
        direct = subdir.split('/')[-1]

        if direct == '':
            continue
        # keys[direct] = OrderedDict()
        for file in files:
            if file=='.DS_Store':
                continue
            file_path = subdir + os.path.sep + file
            shakes = open(file_path, 'r')
            text = shakes.read()
            # lowercasing and taking out the punctuation
            lowers = text.lower()
            # storing the no_token version of each text file (data point or X)
            text_list.append(lowers)

            # storing the index for the file based on directory (therefore U, P, Q) and then based
            # on file which is data name
            # ke/ys[direct][file] = counter

            idx_list.append(direct)
            # print (direct, file)
            counter += 1
    return idx_list, text_list

def tf_idf_mat_calculator(X_train):
    # used ordered dict to keep track of document index stored in token_dict and keys

    # this can take some time... generating the tf-idf matrix
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    # training_set, test_set, train_idx, test_idx = train_test_split(token_dict.values(), np.arange(token_dict.values().shape[-1]))
    tf_idf_fitted_transformer = tfidf.fit(X_train)
    return tf_idf_fitted_transformer

def feature_separator(x):
    x = x.strip()
    feature_split = x.split('\n')
    AC_feature = feature_split[0]
#     print (feature_split)
    remaining_features, TM_feature = feature_split[1:-1], feature_split[-1]
#     remaining_features.reverse()
    x_remaining = '\n'.join(remaining_features)
#     print ("remaining:", x_remaining)
    feature_split = x_remaining.split('-!-')
    if len(x_remaining.split('"')) == 1:
        name_feature = feature_split[0]
    else:
        name_feature = x_remaining.split('"')[0]
    references_feature = ''.join(x_remaining.split('"')[1:-1])
    if len(feature_split) == 1:
        name_feature = x_remaining.split('"')[0]
        comments_feature = ""
    else:
        comments_feature = '\n'.join(feature_split[1:-1] + [feature_split[-1].split('\n')[0]])
        GO_plus_keywords_split = x_remaining.split('"')[-1]
        GO_plus_keywords = '\n'.join(GO_plus_keywords_split)
        GO_plus_keywords_split = GO_plus_keywords.split('GO:')
        
    GO_plus_keywords_split = feature_split[-1].split('\n')[1:]
    GO_plus_keywords = '\n'.join(GO_plus_keywords_split)
    GO_plus_keywords_split = GO_plus_keywords.split('GO:')
    if len(GO_plus_keywords_split) == 1:
        GO_feature = ' '
        keywords_feature = GO_plus_keywords_split[0]
    else:
        GO_feature = '\n'.join(GO_plus_keywords_split[1:-1] + [GO_plus_keywords_split[-1].split('\n')[0]])
        keywords_feature = '\n'.join(GO_plus_keywords_split[-1].split('\n')[1:])
    
    return {'AC':AC_feature.strip(), 'name': name_feature.strip(), 'references': references_feature.strip(), 'comments': comments_feature.strip(), 'GO': GO_feature, 'keywords': keywords_feature.strip(), 'TM': TM_feature.strip()}


if __name__ == "__main__":
    s = "salam. naji"
    import string
    import re
    s = "string.....With. . . . Punctuation"
    table = str.maketrans(".", " ")
    new_text = s.translate(table)
    removed_space = re.sub(' +', ' ', new_text)
    print (removed_space)
    # print (s.translate(table, string.punctuation))
    
