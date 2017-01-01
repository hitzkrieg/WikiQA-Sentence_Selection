
from __future__ import print_function
from functools import reduce
import re
import tarfile
import os
import numpy as np
np.random.seed(1337)  # for reproducibility



from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (precision_score, recall_score,
f1_score, accuracy_score)
import tensorflow as tf
tf.python.control_flow_ops = tf

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 100
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 10
EPOCHS = 1
nb_classes=2
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
GLOVE_DIR= 'Glove'



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def buildtrainfile(fileName):
        fp=open(fileName,'r')
        data= []
        for line in fp:
            #print (line)
            if '\t' in line:
                arr= line.split('\t')
                data.append((tokenize(arr[0]),tokenize(arr[1]),arr[2].strip()))
        fp.close()
        #print (data)
        return data



def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(2)  # let's not forget that index 0 is reserved
        if answer==0:
        	y[0]=1
        else:
        	y[1]=1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def prepare_embeddings_matrix(vocab_size, vocab):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((vocab_size, EMBED_HIDDEN_SIZE))
    for i, word in enumerate(vocab):
        print(word, i)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector    
    return(embedding_matrix)







train = buildtrainfile('WikiQA-train.txt')
#print(data)
test= buildtrainfile('WikiQA-test.txt')
valid= buildtrainfile('WikiQA-dev.txt')

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q ) for story, q, answer in train + test + valid)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
vX, vXq, vY = vectorize_stories(valid, word_idx, story_maxlen, query_maxlen)

Y_label=Y
tY_label=tY
vY_label=vY

Y= np_utils.to_categorical(Y, nb_classes)
tY= np_utils.to_categorical(tY, nb_classes)
vY= np_utils.to_categorical(vY, nb_classes)

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

embedding_matrix= prepare_embeddings_matrix(vocab_size, vocab)
print(embedding_matrix.shape)




print('Build model...')

sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, weights= [embedding_matrix],input_length=story_maxlen, trainable= False))

sentrnn.add(Dropout(0.3))

qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, weights= [embedding_matrix],input_length=query_maxlen, trainable= False))

qrnn.add(Dropout(0.3))
qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
qrnn.add(RepeatVector(story_maxlen))

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)

print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print('2nd method')

y_pred = model.predict_classes([tX, tXq])
read_list(y_pred)
proba = model.predict_proba([tX, tXq])
read_list(proba)
y_test= tY_label
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy: {}'.format(accuracy))
print('Recall: {}'.format(recall))
print('Precision: {}'.format(precision))
print('F1: {}'.format(f1))


model.summary()
