from tensorflow.keras.models import load_model
import numpy as np
import os
import regex as re
import pandas as pd

labels_test = [x.split('\t')[0] for x in open('data/test_62k.txt').readlines()]
labels_test = list(map(int, labels_test))
labels_test = np.array(labels_test)

test_tensor = np.load('tensors/test_tensor_62k.npy')

folder_models = 'models'
data = []
for model in os.listdir(folder_models):
    file_model = folder_models+'/'+model
    model = load_model(file_model)

    nn = re.search("(?<=/).*(?=_)", file_model).group()
    size = re.search("(?<=_).*(?=\.hdf5)", file_model).group()
    row = (nn, size, model.evaluate(test_tensor,labels_test)[1])
    data.append(row)
    
df = pd.DataFrame(data, columns=['nn', 'size', 'acc'])
df.to_csv('model_evaluation.csv', index = False, sep = ';')