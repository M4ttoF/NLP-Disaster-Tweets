import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tokenization
from sklearn.model_selection import train_test_split


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
X = np.array(train['text'])
Y = np.array(train['target'])

test_data = np.array(test['text'])

# BERT LAYER
def bertEncode(texts, tokenizer, maxLen):
    allTokens=[]
    allMasks=[]
    allSegments=[]

    for t in texts:
        t=tokenizer.tokenize(t)
        t=t[:max_length:2]
        inputSequence=['[CLS]']+t+["[SEP]"]

        tokens=tokenizer.convert_tokens_to_ids(inputSequence)
        padLen=maxLen-len(inputSequence)
        tokens+=[0]*padLen
        padMask=[1]*len(inputSequence)+[0]*padLen
        segmentID=[0]*maxLen

        allTokens.append(tokens)
        allMasks.append(padMask)
        allSegments.append(segmentID)

    return np.array(allTokens), np.array(allMasks), np.array(allSegments)



# BUILDING MODEL
def buildModel(bertLayer, maxLen=512):
    
    inputWordIDs=tf.keras.layers.Input(shape=(maxLen,),dtype=tf.int32,name="inputWordIDs")
    inputMask = tf.keras.layers.Input(shape=(maxLen,),dtype=tf.int32,name="inputMask")
    inputSegmentIDs = tf.keras.layers.Input(shape=(maxLen,),dtype=tf.int32,name="inputSegmentIDs")

    _,sequence_output = bertLayer([inputWordIDs,inputMask,inputSegmentIDs])
    clf_output = sequence_output[:,0,:]
    model_X = tf.keras.layers.Dense(100,activation='relu')(clf_output)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    model_X = tf.keras.layers.Dense(100,activation='relu')(model_X)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    model_X = tf.keras.layers.Dense(100,activation='relu')(model_X)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    out = tf.keras.layers.Dense(1,activation='sigmoid')(model_X)

    model = tf.keras.models.Model(inputs=[inputWordIDs,inputMask,inputSegmentIDs],outputs=out)

    return model
moduleUrl = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bertLayer = hub.KerasLayer(moduleUrl, trainable=True)



vocabFile = bertLayer.resolved_object.vocab_file.asset_path.numpy()
lowerCase = bertLayer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocabFile, lowerCase)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1)
train_input = bertEncode(X_train,tokenizer,maxLen=264)
val_input = bertEncode(X_test,tokenizer,maxLen=264)
test_input = bertEncode(test.text.values,tokenizer,maxLen=264)
train_labels = y_train
val_labels = y_test
model = buildModel(bertLayer,maxLen=264)
#print(model.summary())



model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
train_history = model.fit(
    train_input, train_labels,
    validation_data=(val_input,val_labels),
    epochs=6,
    batch_size=16
)

y_pred = model.predict(test_input)

ans = pd.DataFrame({'id':np.array(test['id']),'target':np.array(y_pred.round().astype(int)).reshape(-1)})
ans.to_csv('submission.csv',index=False)
