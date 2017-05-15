import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adadelta
from keras.preprocessing.text import Tokenizer
from model import create_model
from os import path, makedirs, environ
from preprocessing_imdb import *
from w2v import get_embeddings, get_embedding_matrix
import gc

# deactivate tensorflow warnings
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
configuration
'''
outdir = "./models/"
save_base_name = "test_model"
sentences_per_doc = 20
sentence_len = 40
patience = 100
batch_size = 60
epochs = 100
max_vocab_words = None
w2v_file = "./data/GoogleNews-vectors-negative300.bin"

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
train_texts, train_labels = read_file("./data/train_small.txt")
test_texts, test_labels = read_file("./data/test.txt")
valid_texts, valid_labels = read_file("./data/valid.txt")

'''
build vocab and train tokenizer
'''
tokenizer = Tokenizer(num_words=max_vocab_words, lower=True, split=" ")
tokenizer.fit_on_texts(train_texts+valid_texts+test_texts)
embeddings, embedding_dim = get_embeddings(w2v_file, vocab=tokenizer.word_index)
embedding_matrix = get_embedding_matrix(embeddings, embedding_dim, tokenizer.word_index)
print("Embedding matrix shape:", embedding_matrix.shape)
print("Successfully built vocab and generated embedding matrix...")
print(embedding_matrix.shape)
print(embedding_matrix)
'''
build datasets
'''
labels = {}

train_instances, labels = load_sentences(train_texts, train_labels, labels, tokenizer, sentence_len, sentences_per_doc)
valid_instances, labels = load_sentences(valid_texts, valid_labels, labels, tokenizer, sentence_len, sentences_per_doc)
test_instances, labels = load_sentences(test_texts, test_labels, labels, tokenizer, sentence_len, sentences_per_doc)
print('Loaded text data')

'''
train the model
'''
# create the model
model = create_model(embedding_matrix, sentences_per_doc, sentence_len, kernel_sizes=[2, 3, 4], filters=len(labels))
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
model.compile(loss='hinge', optimizer=optimizer, metrics=['accuracy'])
model.fit(
    train_instances.x,
    train_instances.y,
    batch_size,
    epochs,
    verbose=1,
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
