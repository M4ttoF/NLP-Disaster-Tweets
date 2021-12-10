import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Assigning files to variables
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

vocab_size = 10000
max_length = 100

# Get rid of null value columns (keyword and location)
train_data=train.drop('keyword',axis=1)
train_data=train_data.drop('location',axis=1)

# Define X/Y_train data
Y_train=train_data.target
X_train=train_data.text

# Reshape Y_train so LSTM can process it better (number between -1,1)
Y_train=tf.reshape(Y_train,(-1,1))

# Tokenizing the data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_length)


# Declaring model parameters
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[max_length]),
    tf.keras.layers.Embedding(vocab_size,128,input_length = max_length),  

    tf.keras.layers.LSTM(200, return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(200,return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(200),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1,activation='sigmoid') 
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

# TRAINING MODEL HERE
num_epochs = 20
history = model.fit(sequences_matrix,Y_train,batch_size=64,epochs=num_epochs)

# Train accuracy
model.evaluate(sequences_matrix,Y_train)

#Tokenize test data
X_test = test.text
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_test)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_matrix_test = sequence.pad_sequences(sequences_test,maxlen=max_length)

# Prediction on test data
prediction = model.predict(sequences_matrix_test)

# Output to submission.csv
prediction = (prediction>0.5)*1
p=pd.DataFrame()
p['id']=test['id']
p['target']=prediction
p.to_csv('LSTMsubmission.csv',index=False)



