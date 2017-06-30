import argparse
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from os.path import join
from model import create_model
from os import path, makedirs, environ
from preprocessing_imdb import *
from w2v import get_embeddings
import gc

# deactivate tensorflow warnings
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
argument parser
'''
parser = argparse.ArgumentParser(description='CNN classification model for sentences')
parser.add_argument('--train', help='Training file', default="./data/train.txt")
parser.add_argument('--valid', help='Validation file', default="./data/valid.txt")
parser.add_argument('--test', help='Test file', default="./data/test.txt")
parser.add_argument('--embed', help='Word2Vec embeddings file', default="./data/english_300dim.model")
parser.add_argument('--optim', help='Optim method', default='adam', choices=['adam', 'adagrad', 'adadelta', 'sgd',
                                                                             'rmsprop'])
parser.add_argument('--dropout', help='Dropout probability', default=0.5, type=float)
parser.add_argument('--unif', help='Initializer bounds for embeddings', default=0.25)
parser.add_argument('--epochs', help='Number of epochs', default=100, type=int)
parser.add_argument('--batchsz', help='Batch size', default=50, type=int)
parser.add_argument('--sentence_len', help='Max sentences per document', default=35, type=int)
parser.add_argument('--patience', help='Patience', default=50, type=int)
parser.add_argument('--filtersz', help='Filter sizes', nargs='+', default=[3, 4, 5], type=int)
parser.add_argument('--outdir', help='Output directory', default='./models/')
parser.add_argument('--save_name', help='Output Save Base Name', default='test_model')
parser.add_argument('--w2v_dim', help='Set if dimensions of w2v should be scaled down', type=int, default=None)
parser.add_argument('--mx_doc_len', help='Max sentences per document', default=15, type=int)
args = parser.parse_args()

'''
configuration
'''
outdir = args.outdir
save_base_name = args.save_name
sentence_len = args.sentence_len
patience = args.patience
batch_size = args.batchsz
epochs = args.epochs
max_vocab_words = None
sentences_per_doc = args.mx_doc_len
filtersz = args.filtersz
w2v_file = args.embed
w2v_dim = args.w2v_dim
train_file = args.train
test_file = args.test
valid_file = args.valid
optim = args.optim

'''
outdir
'''
if path.exists(outdir) is False:
    print('Creating path: %s' % outdir)
    makedirs(outdir)

'''
load data
'''
print("load data and build vocabulary")
train_texts, train_labels, train_filenames = read_file(train_file)
test_texts, test_labels, test_filenames = read_file(test_file)
valid_texts, valid_labels, valid_filenames = read_file(valid_file)

'''
build vocab and train tokenizer
'''
tokenizer = Tokenizer(num_words=max_vocab_words, lower=True, split=" ")
tokenizer.fit_on_texts(train_texts+valid_texts+test_texts)
embeddings, embedding_dim = get_embeddings(w2v_file, vocab=tokenizer.word_index,
                                           dimension_reduction=w2v_dim)

'''
build datasets
'''
labels = {}

train_instances, labels = load_sentences(train_texts, train_labels, labels, tokenizer, sentence_len, sentences_per_doc,
                                         embeddings)
valid_instances, labels = load_sentences(valid_texts, valid_labels, labels, tokenizer, sentence_len, sentences_per_doc,
                                         embeddings)
test_instances, labels = load_sentences(test_texts, test_labels, labels, tokenizer, sentence_len, sentences_per_doc,
                                        embeddings)
print('Loaded text data')

'''
train the model
'''
# create the model
model = create_model(embedding_dim, sentences_per_doc, sentence_len, kernel_sizes=filtersz, filters=100)
print("")
print('=====================================================')
print("Model Summary:")
print(model.summary())
print('=====================================================')

# prepare some functions
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
outname = join(outdir, save_base_name+".model")
checkpoint = ModelCheckpoint(outname, monitor='val_loss', verbose=1, save_best_only=True)

# run training
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
model.fit(
    train_instances.x,
    train_instances.y,
    batch_size,
    epochs,
    verbose=2,
    callbacks=[checkpoint, early_stopping],
    validation_data=(valid_instances.x, valid_instances.y),
    shuffle=True
)

'''
evaluation
'''
print("")
print('=====================================================')
print('Evaluating best model on test data:')
print('=====================================================')
model = load_model(outname)
start_time = time.time()
score = model.evaluate(test_instances.x, test_instances.y, batch_size, verbose=1)
duration = time.time() - start_time

print('Test (Loss %.4f) (Acc = %.4f) (%.3f sec)' % (score[0], score[1], duration))

'''
https://github.com/tensorflow/tensorflow/issues/3388#issuecomment-268502675
'''
gc.collect()
