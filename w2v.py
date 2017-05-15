import numpy as np


def readtospc(f):
    s = bytearray()
    ch = f.read(1)

    while ch != b'\x20':
        s.extend(ch)
        ch = f.read(1)
    s = s.decode('utf-8')
    return s.strip()


def get_embeddings(filename, vocab=None):
    """
    
    :param filename: 
    :param vocab: 
    :return: 
     - embeddings: dictionary, where the key is the index of a word from the vocab, the value is the word vector
     - embedding_dim: dimensions of the wordvector (int)
    """
    embeddings_index = {}

    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, embedding_dim = map(int, header.split())

        vector_width = 4 * embedding_dim

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
            embeddings_index[vocab[word]] = vector

        # print vocab info
        print("With word2vec matched words:", vocab_matched_words)
        print("With word2vec unmatched words:", vocab_unmatched_words)

    return embeddings_index, embedding_dim


def get_embedding_matrix(embeddings_index, embedding_dim, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    unif_vec = np.random.uniform(low=0.0, high=1.0, size=embedding_dim)

    for word, idx in word_index.items():
        embedding_matrix[idx] = embeddings_index.get(word, unif_vec)[:embedding_dim]

    return embedding_matrix
