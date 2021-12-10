import numpy as np
from numpy.core.records import array
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Assigning files to variables
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

# Define variables to be set for training the model
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

#
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

# Convert padded data to np array
training_padded = np.array(train_padded)
training_labels = np.array(y_train)
testing_padded = np.array(valid_padded)
testing_labels = np.array(y_valid)

# TRAINING MODEL HERE
num_epochs = 1

n_members = 10

models = list()
for i in range(n_members):
    print("Trained Model# " + str(i))
    model = model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(
        training_padded, training_labels, 
        epochs=num_epochs*i, 
        validation_data=(testing_padded, testing_labels), 
        verbose=0)
    models.append(model)

# Tokenizing data
test_seq = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(
    test_seq, maxlen=max_length, padding=padding_type,
    truncating=trunc_type)

# Get predictions from each model
yhats = [model.predict(test_padded) for model in models]
yhats = np.array(yhats)
# average each model's prediction for each input
averaged = np.average(yhats, axis = 0)
# change values to either 1 or 0
outcomes = np.where(averaged > 0.5, 1, 0)


# # Saving predictions to csv
output = test['id']
output = pd.DataFrame(output)
output['target'] = outcomes
output.head()
output.to_csv('EMsubmission.csv', index=False)
output = np.array(output)

correct = pd.read_csv("socialmedia-disaster-tweets-DFE.csv")
correct = correct[['choose_one', 'text']]
correct['target'] = (correct['choose_one'] == 'Relevant').astype(int)
correct['id'] = correct.index

merged = pd.merge(test,correct, on="id")

subm = merged[['id','target']]

submnp = np.array(subm)

sum = 0

# Calculate accuracy of the ensemble
for i in range(submnp.shape[0]):
    if(submnp[i][1] == output[i][1]):
        sum += 1
print("Accuracy = " + str(sum/submnp.shape[0]))