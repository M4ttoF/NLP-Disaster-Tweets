import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Assigning files to variables
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 5000

# Saving training/testing data to variables
train_data = train['text'].tolist()
train_labels = train['target'].tolist()

test_data = test['text'].tolist()

# Only train/test for the previously set training size
X_train = train_data[:training_size]
X_valid = train_data[training_size:]

y_train = train_labels[:training_size]
y_valid = train_labels[training_size:]



tokenizer = Tokenizer(num_words=vocab_size, oov_token= oov_tok)
tokenizer.fit_on_texts(X_train)


train_seq = tokenizer.texts_to_sequences(X_train)
valid_seq = tokenizer.texts_to_sequences(X_valid)


# Pads training data for any rows missing id,keyword,location
train_padded = pad_sequences(
    train_seq, maxlen=max_length, padding=padding_type,
    truncating=trunc_type)

valid_padded = pad_sequences(
    valid_seq, maxlen=max_length, padding=padding_type,
    truncating=trunc_type)



# Declaring model parameters
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# Convert padded data to np array
training_padded = np.array(train_padded)
training_labels = np.array(y_train)
testing_padded = np.array(valid_padded)
testing_labels = np.array(y_valid)

# TRAINING MODEL HERE
num_epochs = 30
history = model.fit(
    training_padded, training_labels, 
    epochs=num_epochs, 
    validation_data=(testing_padded, testing_labels), 
    verbose=2)

# Tokenizing data
test_seq = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(
    test_seq, maxlen=max_length, padding=padding_type,
    truncating=trunc_type)



# Setting Predictions
predicted_data = model.predict(test_padded).flatten() 
predicted = pd.DataFrame(predicted_data, columns=['target'])

# Prediction values are floats so we have to round for output
predicted['target'] = np.where(predicted['target'] > 0.5 , 1, 0)
predicted.head()

# Saving predictions to csv
output = test['id']
output = pd.DataFrame(output)
output['target'] = predicted
output.head()

output.to_csv('submission.csv', index=False)