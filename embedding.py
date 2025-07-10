from keras.layers import TextVectorization,Input
import tensorflow as tf

sent=[
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good'
]

print(sent)

## define the vocabulary size

vocab_size=10000

##one hot representation for every word
texts=tf.constant(sent)
vectorizer=TextVectorization(
    max_tokens=vocab_size,
    output_mode='int'
)

vectorizer.adapt(texts)

one_hot_output=vectorizer(texts)

print(one_hot_output)

## word embedding reoresentation

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

import numpy as np

sent_length=8

embedded_docs=pad_sequences(one_hot_output,padding='pre',maxlen=sent_length)
print(embedded_docs)

## feature representation

dim=10
model=Sequential()
model.add(Input(shape=(sent_length,)))
model.add(Embedding(input_dim=vocab_size,output_dim=dim))
model.compile('adam','mse')

import io
from contextlib import redirect_stdout
stream=io.StringIO()
with redirect_stdout(stream):
    model.summary()
summary_str=stream.getvalue()
print("s",summary_str)

print(model.predict(np.array([embedded_docs[0]])))

