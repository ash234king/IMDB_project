import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense,Input

## load the imdb dataset
max_features=10000 ## vocabulary size
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
print(f'Training data shape: {x_test.shape}, Training labels shape: {y_test.shape}')

## inspect a sample review and its label
sample_review=x_train[0]
sample_label=y_train[0]

print(f"Sample review (as integers): {sample_review}")


## maping of words index bacl to words
word_index=imdb.get_word_index()
print(word_index)

reverse_word_index={value: key for key,value in word_index.items()}

print(reverse_word_index)

decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

from keras.preprocessing import sequence
max_len=500

x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)

print(x_train)

## train simple rnn
model=Sequential( )
model.add(Input(shape=(max_len,)))
model.add(Embedding(input_dim=max_features, output_dim=128))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print("model ",model.summary())

## create an instance of EarlyStopping Callback
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

## train the model with early stopping
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,callbacks=[early_stopping])


model.save('models/simple_rnn.keras')


