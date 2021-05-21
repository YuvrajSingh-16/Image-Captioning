#!/usr/bin/env python
# coding: utf-8

# ## Caption IT

# In[1]:


## Library imports
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
# from keras.models import Model, load_model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
import json
import pickle
from collections import Counter

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# In[3]:


model = load_model("./model_weights/final_model_19.h5")


# In[4]:


tmp_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))


# In[5]:


## Removing the last layer from the model
resnet_model = Model(tmp_model.input, tmp_model.layers[-2].output)


# In[6]:


## Image preprocessing functions
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224, 3))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0) 
    img = preprocess_input(img)
    
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet_model.predict(img)
    feature_vector = feature_vector.reshape((1, feature_vector.shape[1]))
    return feature_vector


# In[7]:


img = encode_image("test_images/bike.jpg")
img


# In[8]:


## Importing descriptions dictionary
with open("descriptions.txt", "r") as f:
    descriptions = f.read()
    descriptions = json.loads(descriptions.replace("\'", "\""))

imgs = list(descriptions.keys())
train_data = imgs[:6092]


# In[9]:


## Loading word2idx & idx2word
word_to_idx = None
idx_to_word = None

with open("word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)

vocab_size = len(word_to_idx) + 1  ## plus one to keep zero index for padding and start 


# In[10]:


max_len = 31

def predict_caption(photo):
    in_text = 'startseq'
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final




