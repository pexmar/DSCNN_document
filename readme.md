# Dependency Sensitive Convolutional Neural Networks
Implementation of the DSCNN for document modeling from the paper [Dependency Sensitive Convolutional Neural Networks for Modeling Sentences and Documents](https://arxiv.org/abs/1611.02361) (NAACL 2016).
In this implementation the use of multiple embeddings was left out.

The code for modeling sentences was published by the author here: https://github.com/ryanzhumich/dscnn 

Demo with a small extract of [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) dataset.

## Dependencies
- Python (3.5)
- Keras (2.0)

## Prepare Pretrained Word Embeddings

The model uses preptrained word embeddings including [word2vec](https://code.google.com/archive/p/word2vec/).
Download those word embeddings and save them as: data/GoogleNews-vectors-negative300.bin

## Data Preprocessing

The network expects the dataset splitted into train-, validation and testset as TSV in the format (splitted by '\t)

```
class_id	Text
```

## Configuration

In the classify.py you will find a section with configuration variables:

```
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
```

Here you can configure the network.

There are some more possibilities to configure the network that you will find also in the classify.py (e.g. Optimizer, Loss-function etc.)

## Run Demo

You can simply run the classify.py:

```
python3 classify.py 
```

You can run the Demo also on your GPU. Depending on which backend your Keras is using, tensorflow will detect the GPU automatically, but for Theano you can use THEANO-Flags:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3 classify.py
```
