import numpy as np


def readtospc(f):
    s = bytearray()
    ch = f.read(1)

    while ch != b'\x20':
        s.extend(ch)
        ch = f.read(1)
    s = s.decode('utf-8')
    return s.strip()


def get_embeddings(filename, vocab=None, dimension_reduction=None, init_unknown_words=True):
    """

    :param init_unknown_words:
    :param filename:
    :param vocab:
    :param dimension_reduction: dimension to which the word vectors should be scaled down
    :return:
     - embeddings: dictionary, where the key is the index of a word from the vocab, the value is the word vector
     - embedding_dim: dimensions of the wordvector (int)
    """
    embeddings_index = {}

    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, embedding_dim = map(int, header.split())

        vector_width = 4 * embedding_dim

        # dimension reduction
        if dimension_reduction:
            embedding_dim = dimension_reduction
        else:
            dimension_reduction = embedding_dim

        # count matched vocab
        vocab_matched_words = 0
        vocab_unmatched_words = 0

        for vocab_idx in range(vocab_size):
            word = readtospc(f)
            raw = f.read(vector_width)

            if word not in vocab:
                vocab_unmatched_words += 1
                continue
            else:
                vocab_matched_words += 1

            vector = np.fromstring(raw, dtype=np.float32)

            if dimension_reduction:
                vector = vector[:dimension_reduction]

            if sentiws is not None:
                vector = np.append(vector, sentiws.get(word, 0))

            embeddings_index[vocab[word]] = vector

        # print vocab info
        print("With word2vec matched words:", vocab_matched_words)
        print("With word2vec unmatched words:", vocab_unmatched_words)

    embeddings_index[0] = np.random.uniform(-0.25, 0.25,
                                            dimension_reduction if sentiws is None else dimension_reduction + 1)

    return embeddings_index, embedding_dim


def get_embedding_matrix(embeddings_index, embedding_dim, word_index):
    """
    optional, if you want to use the embedding layer in the model.
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    unif_vec = np.random.uniform(low=0.0, high=1.0, size=embedding_dim)

    for word, idx in word_index.items():
        embedding_matrix[idx] = embeddings_index.get(word, unif_vec)[:embedding_dim]

    return embedding_matrix
