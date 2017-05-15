import json
import re
from collections import Counter
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
from os.path import join


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def read_file(file):
    texts = []  # list of text samples
    labels = []  # list of labels

    with open(file, 'r') as f:
        for l in f:
            line = l.split()

            # labeling
            label_id = line[0]

            # texts
            texts.append(" ".join(line[1:]))  # TODO: wenn filename dazwischen ist, entsprechend line[2:]
            labels.append(label_id)

    print('Found %s texts.' % len(texts))
    return texts, labels


def splits(text):
    return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))


def sentence_tokenizer(text):
    """
    Tokenizes sentences.

    :param text:
    :return: list of sentences (a sentence is a string)
    """

    punkt_param = PunktParameters()
    punkt_param.abbrev_types = {'zzgl', 'prof', 'ca', 'vj', 't', 'mio', 'sro', 'lv', 'io', 'ihv', 'bzw', 'usw', 'inkl',
                                'zt', 'vh', 'dr', 'entspr', 'dem', 'fort', 'co', 'kg', 'zb', 'bspw', 'ua', 'rd', 'abs',
                                'etc', 'tsd', 'z.b', 'evtl', '1', '2', '3', '4', '5', '6', '7', '8', '9', '19', '20',
                                '21'}
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    return sentence_splitter.tokenize(text)


def build_vocab(text_instances):
    vocab = Counter()
    for inst_text in text_instances:
        for w in splits(inst_text):
            vocab[w] += 1
    return vocab


class SentenceLabelExamples(object):

    def __init__(self, x, y):
        self.x = x
        self.y = to_categorical(y, num_classes=2)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def width(self):
        return self.x[0].shape[0]


def load_sentences(inst_texts, inst_labels, labels, tokenizer, max_sentence_len, max_sentences_per_doc):
    """
    This method will return a list (x) that contains a list for each instance. One instance contains a list of 
    sentences. Each sentence is one np.array of length mxlen, with one index for each word.     

    :param inst_texts: 
    :param inst_labels: 
    :param labels:
    :param tokenizer:
    :param max_sentence_len: 
    :param max_sentences_per_doc: 
    :return: x is list (each item is one instance) of list (each item is a sentence) of np-arrays (sequence of word ids)
    """
    number_instances = len(inst_texts)
    inputs = [np.zeros((number_instances, max_sentence_len)) for _ in range(max_sentences_per_doc)]
    y = np.zeros(number_instances, dtype=int)

    for inst_idx in range(number_instances):
        '''
        label
        '''
        if not inst_labels[inst_idx] in labels:
            labels[inst_labels[inst_idx]] = len(labels)
        y[inst_idx] = labels[inst_labels[inst_idx]]

        '''
        text
        '''
        # get list of all sentences and clean them
        sentences = sentence_tokenizer(inst_texts[inst_idx])
        sentences = [clean_str(text) for text in sentences]

        # sequence and pad each sentence
        sentences = tokenizer.texts_to_sequences(sentences)
        sentences = list(pad_sequences(sentences, maxlen=max_sentence_len))

        # pad sentence length
        for input_idx in range(min(max_sentences_per_doc, len(sentences))):
            inputs[input_idx][inst_idx] = sentences[input_idx]

    return SentenceLabelExamples(inputs, y), labels


@DeprecationWarning
def mdsave(labels, vocab, outdir, save_base):
    basename = join(outdir, save_base)

    label_file = basename + '.labels'
    print("Saving attested labels '%s'" % label_file)

    with open(label_file, 'w') as f:
        json.dump(labels, f)

    vocab_file = basename + '.vocab'
    print("Saving attested vocabulary '%s'" % vocab_file)
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
