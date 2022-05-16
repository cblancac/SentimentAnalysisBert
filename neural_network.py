from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D, concatenate, Embedding
from tensorflow.keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, GlobalMaxPooling1D, LSTM


def get_cnn_model(input_shape,
                  n_neurons=50,
                  dropout_rate=0.1,
                  num_filter=256,
                  num_words=3,
                  opt_alg='nadam'):

    inp = Input(shape=input_shape)
    x = Conv1D(filters=num_filter,
                 kernel_size=num_words,
                 activation="relu")(inp)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_alg,
                  metrics=['acc'])

    return model


def get_bigru_model(input_shape,
                            n_neurons,
                            dropout_rate=0.1,
                            opt_alg='nadam'):
    inp = Input(shape=input_shape)
    x = Bidirectional(GRU(n_neurons, return_sequences=True))(inp)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_alg,
                  metrics=['acc'])

    return model


def get_bigru_cnn_model(input_shape,
                               n_neurons=128,
                               n_filters=64,
                               kernel_size=2,
                               dropout_rate=0.2,
                               lr=1e-3,
                               lr_d=0.0):
    inp = Input(shape=input_shape)
    x = SpatialDropout1D(dropout_rate)(inp)

    x = Bidirectional(GRU(n_neurons, return_sequences=True))(x)
    x = Conv1D(n_filters,
               kernel_size=kernel_size,
               padding="valid",
               kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr, decay=lr_d),
                  metrics=['acc'])
    
    return model


def get_full_bidirectional_model(embed_size_n_feat,   
                                 vocab_size,
                                 input_shape,
                                 n_neurons,
                                 dropout_rate=0.2,
                                 opt_alg='nadam'):
    inp = Input(shape=(input_shape,))
    x = Embedding(vocab_size, embed_size_n_feat)(inp)
    x = Bidirectional(GRU(n_neurons, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_alg,
                  metrics=['acc'])

    return model
    
    
def get_lstm_model(input_shape,
		    n_neurons,
		    dropout_rate=0.1,
		    opt_alg='nadam'):
		    
    inp = Input(shape=input_shape)
    x = LSTM(n_neurons, return_sequences=True)(inp)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_alg,
                  metrics=['acc'])

    return model
