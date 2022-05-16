import numpy as np
import yaml
import os
import pandas as pd

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from neural_network import get_cnn_model, get_lstm_model, get_bigru_model, get_bigru_cnn_model


class TrainModel():

    def __init__(self, X_train, y_train, model):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model
        self.model = self.get_model(model)


    def get_model(self, model):
        """Replace the names of the models given in the input for
           the real names of the models defined in neural_network.py"""

        dictOfModels = {'cnn' : get_cnn_model,
                        'lstm': get_lstm_model,
                        'gru': get_bigru_model,
                        'gru+cnn': get_bigru_cnn_model}
        try:
            model = dictOfModels[model]
            return model
        except:
            print("The model selected is not included in the list")
            quit()
        

    def get_info_by_size(self, N):
        """Select the config and the size of X and Y
           depending on the size given by N"""

        folder_config = "configs/"
        if N <= 1000: path_file = "config_small.yaml"
        elif N <=10000: path_file = "config_medium.yaml"
        else: path_file = "config_large.yaml"

        X, Y = self.X_train[:N], self.y_train[:N]
        
        path_config_file = folder_config + path_file
        with open(path_config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return X, Y, config


    def hyperparameter_optimizer(self, X, Y, config):
        """Train models using different combinations of hyperparameters
           saving the one who offer the best result"""

        create_model = KerasClassifier(build_fn=self.model, input_shape=X.shape[1:])

        n_neurons = [50, 75]
        dropout_rate = [0.25, 0.4, 0.5]
        param_dist = dict(n_neurons=n_neurons, 
                        dropout_rate=dropout_rate)

        n_iter_search = 3 # Number of parameter settings that are sampled.
        random_search = RandomizedSearchCV(estimator=create_model,
                                        param_distributions=param_dist,
                                        n_iter=n_iter_search,
                                        n_jobs=1,
                                        cv = 10,
                                        verbose=1)

        file_path = f"models/{model}_{X.shape[0]}.hdf5"

        early_stop = EarlyStopping(monitor='val_acc',  
                                min_delta=config['es_min_delta'],
                                patience=config['es_patience'], 
                                mode='max',  
                                verbose=0)
        checkpoint = ModelCheckpoint(file_path, 
                                    monitor='val_acc', 
                                    verbose=0, 
                                    mode='max',   
                                    save_best_only=True)
        reduce_lr = ReduceLROnPlateau( monitor='val_acc', 
                                    factor=config['rlr_factor'], 
                                    patience=config['rlr_patience'], 
                                    cooldown=config['rlr_cooldown'], 
                                    min_lr=config['rlr_min_lr'])

        callbacks_list = [checkpoint,
                        early_stop,
                        reduce_lr]

        history = random_search.fit(X, Y,
                        epochs=config['epochs'], 
                        validation_split=config['val_prop'], 
                        callbacks=callbacks_list)

        means = random_search.cv_results_['mean_test_score']
        stds = random_search.cv_results_['std_test_score']
        params = random_search.cv_results_['params']

        return random_search.best_score_, random_search.best_params_, means, stds, params


    def generate_inform(self, best_score, best_params, means, stds, params):
        """Save in csv file some info related with the best model trained"""
        print("Best: %f using %s" % (best_score, best_params))
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        data = [(str(self.model_name), X.shape[0], best_score, best_params['n_neurons'], best_params['dropout_rate'])]
        
        if not os.path.exists('results.csv'):
            df_results = pd.DataFrame(data, columns=['model', 'N', 'best_score', 'n_neurons', 'dropout_rate'])
        else:
            df_results = pd.read_csv('results.csv', sep = ';')
            idxs_model = df_results.index[df_results['model']==data[0][0]].tolist()
            idxs_N = df_results.index[df_results['N']==data[0][1]].tolist()
            if idxs_model and idxs_N and len(list(set(idxs_model) & set(idxs_N)))==1:
                idx = list(set(idxs_model) & set(idxs_N))[0]
                if df_results.loc[idx, 'best_score'] < data[0][2]:
                    df_results.at[idx,'best_score']=data[0][2]
                    df_results.at[idx,'n_neurons']=data[0][3]
                    df_results.at[idx,'dropout_rate']=data[0][4]
            else:
                df_results.loc[len(df_results.index)] = [i for i in data[0]]
        df_results.to_csv('results.csv', index = False, sep = ';')



if __name__ == '__main__':

    X_train = np.load('tensors/train_tensor_150k.npy')
    y_train = [x.split('\t')[0] for x in open('data/train_150k.txt').readlines()]
    y_train = list(map(int, y_train))

    N = int(input('Enter the size of the train dataset: '))
    model = input('Enter your favourite model cnn, lstm, gru or gru+cnn: ')

    t = TrainModel(X_train, y_train, model)
    X, Y, config = t.get_info_by_size(N)
    best_score, best_params, means, stds, params = t.hyperparameter_optimizer(X, Y, config)
    t.generate_inform(best_score, best_params, means, stds, params)