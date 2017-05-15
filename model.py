from keras.layers import Activation, Input, AveragePooling1D, MaxPool1D, Conv1D, concatenate, Dropout, GlobalMaxPool1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model


def create_model(embedding_matrix, max_sentences_per_doc, max_sentence_len, kernel_sizes, filters=2, dropout=0.5):
    """
    
    :param embedding_matrix: 
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
    sentence_inputs = [Input(shape=(max_sentence_len,), name="input_" + str(i)) for i in range(max_sentences_per_doc)]

    # embedding (sentence-level)
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    shared_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                 input_length=max_sentence_len)
    sentence_modeling = [shared_embedding(sentence_inputs[i]) for i in range(max_sentences_per_doc)]

    # LSTMs and Average Pooling (sentence-level)
    shared_sentence_lstm = LSTM(units=embedding_dim, return_sequences=True, activation='tanh')
    shared_average_pooling = AveragePooling1D(pool_size=max_sentence_len)
    sentence_modeling = [shared_sentence_lstm(sentence_modeling[i]) for i in range(max_sentences_per_doc)]
    sentence_modeling = [shared_average_pooling(sentence_modeling[i]) for i in range(max_sentences_per_doc)]

    '''
    document modeling
    '''
    doc_modeling = concatenate(sentence_modeling, axis=1)
    doc_modeling = LSTM(units=embedding_dim, activation='tanh', return_sequences=True)(doc_modeling)

    convolutions = [Conv1D(filters=filters, kernel_size=k, activation='relu')(doc_modeling) for k in kernel_sizes]
    doc_modeling = concatenate(convolutions, axis=1)

    doc_modeling = GlobalMaxPool1D()(doc_modeling)

    doc_modeling = Dropout(dropout)(doc_modeling)
    doc_modeling = Activation('softmax')(doc_modeling)

    return Model(inputs=sentence_inputs, outputs=doc_modeling)
