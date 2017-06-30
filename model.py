from keras.layers import Input, AveragePooling1D, MaxPooling1D, Convolution1D, Concatenate, Dropout, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Model


def create_model(embedding_dim, max_sentences_per_doc, max_sentence_len, kernel_sizes, filters=100, dropout=0.5,
                 hidden_dims=100):
    """
    
    :param hidden_dims:
    :param embedding_dim:
    :param max_sentences_per_doc:
    :param max_sentence_len: 
    :param filters: 
    :param kernel_sizes: 
    :param dropout
    :return: 
    """
    '''
    sentence modeling 
    '''
    # input (sentence-level)
    sentence_inputs = [Input(shape=(max_sentence_len, embedding_dim,), name="input_" + str(i))
                       for i in range(max_sentences_per_doc)]

    # LSTMs and Average Pooling (sentence-level)
    shared_sentence_lstm = LSTM(units=embedding_dim, return_sequences=True, activation='tanh')
    shared_average_pooling = AveragePooling1D(pool_size=max_sentence_len)
    sentence_modeling = [shared_sentence_lstm(sentence_inputs[i]) for i in range(max_sentences_per_doc)]
    sentence_modeling = [shared_average_pooling(sentence_modeling[i]) for i in range(max_sentences_per_doc)]

    '''
    document modeling
    '''
    doc_modeling = Concatenate(axis=1)(sentence_modeling)
    doc_modeling = LSTM(units=embedding_dim, activation='tanh', return_sequences=True)(doc_modeling)

    conv_blocks = []
    for sz in kernel_sizes:
        conv = Convolution1D(filters=filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(doc_modeling)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    doc_modeling = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    doc_modeling = Dropout(dropout)(doc_modeling)
    doc_modeling = Dense(hidden_dims, activation="relu")(doc_modeling)

    model_output = Dense(1, activation="sigmoid")(doc_modeling)

    return Model(inputs=sentence_inputs, outputs=model_output)
