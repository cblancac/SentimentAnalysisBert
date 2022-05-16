import pandas as pd
import numpy as np

class GetInfo:
    
    def __init__(self, texts, pctl):
        self.texts = texts
        self.pctl = pctl

    def get_max_seq_length(self):
        words_texts = [len(e.split()) for e in list(self.texts.tweet)]
        percentile = int(np.percentile(words_texts, self.pctl))
        return percentile

if __name__ == "__main__":
    df_train = pd.read_csv('data/train_150k.txt', sep='\t', names=('sent', 'tweet'))
    pctl = 99.95
    p1 = GetInfo(df_train, pctl)
    print(p1.get_max_seq_length())
    
    
