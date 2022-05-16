import mxnet as mx
from bert_embedding import BertEmbedding
import numpy as np
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from tqdm import tqdm


class BertEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self,
                 mx_use_gpu=True,
                 spacy_nlp_model_name="en_core_web_lg",
                 max_seq_length=300,
                 batch_size=256,
                 sentence_splitting=False,
                 padding='central'):

        self.spacy_nlp_model_name = spacy_nlp_model_name
        self.spacy_nlp = spacy.load(spacy_nlp_model_name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.sentence_splitting = sentence_splitting

        if mx_use_gpu:
            mx_context = mx.gpu(0)
        else:
            mx_context = mx.cpu(0)
        self.mx_context = mx_context
        self.bert_embedder = BertEmbedding(ctx=mx_context,
                                           max_seq_length=max_seq_length,
                                           batch_size=batch_size)
        self.bert_embedding_dim = 768
        self.unknown_representation_arr = np.zeros((max_seq_length, self.bert_embedding_dim),
                                                   dtype=np.float32)

        assert padding in ['central', 'left', 'right'], "Bad padding mode"
        self.padding = padding


    def pad(self, arr):

        arr_size = arr.shape[0]
        n_to_complete = int(self.max_seq_length - arr_size)

        if self.padding == 'right':
            padd_arr = np.pad(arr,
                              pad_width=((0, n_to_complete), (0, 0)),
                              mode='constant',
                              constant_values=0)

        elif self.padding == 'left':
            padd_arr = np.pad(arr,
                              pad_width=((n_to_complete, 0), (0, 0)),
                              mode='constant',
                              constant_values=0)

        elif self.padding == 'central':
            n_left = int(np.floor(n_to_complete / 2))
            if n_to_complete % 2:
                n_right = n_left + 1
            else:
                n_right = n_left

            padd_arr = np.pad(arr,
                              pad_width=((n_left, n_right), (0, 0)),
                              mode='constant',
                              constant_values=0)

        return padd_arr


    def get_bert_array(self, text):

        if not self.sentence_splitting:
            bert_emb = self.bert_embedder([text])
            flat_arr_list = bert_emb[0][1]

        else:
            doc = self.spacy_nlp(text)
            sentences = [sent.text for sent in doc.sents]
            bert_emb = self.bert_embedder(sentences)
            arr_list = [element[1] for element in bert_emb]
            flat_arr_list = reduce(lambda l1, l2: l1 + l2, arr_list)

        if len(flat_arr_list) >= 1:
            emb_arr = np.vstack(flat_arr_list)
        else:
            emb_arr = self.unknown_representation_arr

        emb_arr = self.pad(emb_arr)

        return emb_arr

    def transform(self, text_series):

        emb_arr_list = []

        for text in tqdm(text_series.values, desc="PERFORMING BERT EMBEDDINGS"):
            emb_arr = self.get_bert_array(text)
            emb_arr_list.append(emb_arr)

        tensor = np.array(emb_arr_list)

        return tensor