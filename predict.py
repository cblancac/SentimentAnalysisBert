import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from info_dataset import GetInfo
from bert_keras import BertEmbedder
from translate import Translator

df_train = pd.read_csv('data/train_150k.txt', sep='\t', names=('sent', 'tweet'))

tlt = Translator("en")
info = GetInfo(df_train, 99.95)
MAX_SEQ_LENGTH = info.get_max_seq_length()

bert_embedder = BertEmbedder(False, # False if you don't have gpu, True in another case
                             spacy_nlp_model_name='en_core_web_lg', 
                             max_seq_length=MAX_SEQ_LENGTH, 
                             sentence_splitting=False,
                             padding='left')

# The model choosen is the best trained in terms of accuracy
model = load_model('models/gru+cnn_10000.hdf5')

# je suis heureux | Estaría feliz | سوف أسافر الأسبوع المقبل | 悲しいだろう | I am so sad
sentences = input("Introduce here the sentence(s) you want to predict (separate by | if you introduce more than one):\n ")
sentences = sentences.split('|')
sentences = [s.strip() for s in sentences]

lang_sent_en = [tlt.translate_sentence(sentence) for sentence in sentences]
sentences_en = [i[1] for i in lang_sent_en]

test_emb_arr_list = []
for text in sentences_en:
    emb_arr = bert_embedder.get_bert_array(text)
    test_emb_arr_list.append(emb_arr)

test_tensor = np.array(test_emb_arr_list)
feeling_value = model.predict(test_tensor)

feelings = list(zip(sentences,
                    [{'to_en': i[1]} for i in lang_sent_en],
                    [{'label':'POSITIVE', 'score': j[0]} if j[0]>0.5 else {'label':'NEGATIVE', 'score':1-j[0]} for j in feeling_value]))

feelings = [(i[0],{**i[1], **i[2]}) for i in feelings]
for f in feelings:
    print(f)