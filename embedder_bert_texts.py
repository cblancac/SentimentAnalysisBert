import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm

from bert_keras import BertEmbedder
from info_dataset import GetInfo
nlp = spacy.load('en_core_web_lg')

df_train = pd.read_csv('data/train_150k.txt', sep='\t', names=('sent', 'tweet'))
df_test = pd.read_csv('data/test_62k.txt', sep='\t', names=('sent', 'tweet'))

info = GetInfo(df_train, 99.95)
MAX_SEQ_LENGTH = info.get_max_seq_length()


bert_embedder = BertEmbedder(False, # False if you don't have gpu, True in another case
                             spacy_nlp_model_name='en_core_web_lg', 
                             max_seq_length=MAX_SEQ_LENGTH, 
                             sentence_splitting=False,
                             padding='left')


def get_tensor_subset(df, dst):
    """ Takes a text series, outputs a tensor (using BERT)"""
    emb_arr_list = []
    for text in tqdm(df.tweet.values):
        emb_arr = bert_embedder.get_bert_array(text)
        emb_arr_list.append(emb_arr)

    tensor = np.array(emb_arr_list)
    np.save(dst, tensor)



if __name__ == "__main__":

    subset = input('Enter your subset here (train or test): ')

    if subset == "train":
        get_tensor_subset(df_train, 'tensors/train_tensor_150k.npy')
    elif subset == "test":
        get_tensor_subset(df_test, 'tensors/test_tensor_62k.npy')
    else:
        print("The subset selected is not correct")