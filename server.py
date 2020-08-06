#!/usr/bin/env python3
 
"""Simple HTTP Server With Upload.

For hosting swiftlatex locally.
"""


import os
import posixpath
import http.server
import urllib.request, urllib.parse, urllib.error
import shutil
import mimetypes
import re
import json
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, LSTM, Flatten, TimeDistributed, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K

import unicodedata
import re
import numpy as np
import os
import time
import shutil

import pandas as pd
import numpy as np
import string, os 
print(tf.__version__)

model=tf.keras.models.load_model('my_model.h5')
encoder_model=tf.keras.models.load_model('encoder.h5')
inf_model=tf.keras.models.load_model('inf_model.h5')
file = open("./dataset.txt", 'r')
corpus = [line.strip() for line in file]

def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, '')
    return text

      
def preprocess(data):
    output = []
    punct = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
    for line in data:
         pline= clean_special_chars(line.lower(), punct)
         output.append(pline)
    return output  


def generate_dataset():
  
    processed_corpus = preprocess(corpus)    
    output = []
    for line in processed_corpus:
        token_list = line
        for i in range(1, len(token_list)):
            data = []
            x_ngram = '<start> '+ token_list[:i+1] + ' <end>'
            y_ngram = '<start> '+ token_list[i+1:] + ' <end>'
            data.append(x_ngram)
            data.append(y_ngram)
            output.append(data)
    print("Dataset prepared with prefix and suffixes for teacher forcing technique")
    dummy_df = pd.DataFrame(output, columns=['input','output'])
    return output, dummy_df            
    
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word

def max_length(t):
    return max(len(i) for i in t)

def load_dataset():
    pairs,df = generate_dataset()
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_length(input_data), max_length(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")
    return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out, df

input_data, teacher_data, input_lang, target_lang, len_input, len_target, df = load_dataset()
target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

# Shuffle all of the data in unison. This training set has the longest (e.g. most complicated) data at the end,
# so a simple Keras validation split will be problematic if not shuffled.

p = np.random.permutation(len(input_data))
input_data = input_data[p]
teacher_data = teacher_data[p]
target_data = target_data[p]

# Converts the given sentence (just a string) into a vector of word IDs
# Output is 1-D: [timesteps/words]

def sentence_to_vector(sentence, lang):

    pre = '<start> '+ sentence + ' <end>'
    vec = np.zeros(len_input,dtype=np.int32)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

def vector_to_sentence(vec, lang):
    sentence=' '.join([lang.idx2word[x] for x in vec if x>0 and x!=lang.word2idx['<start>'] and x!=lang.word2idx['<end>']])
    return sentence

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel):
    sv = sentence_to_vector(input_sentence, input_lang)
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]
    
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""

    while cur_word != "<end>" and i < (len_target-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        x_in = [cur_vec, sh, sc]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        cur_vec[0,0] = np.argmax(nvec[0,0])
        cur_word = target_lang.idx2word[np.argmax(nvec[0,0])]
    return output_sentence

def testf(index):
    print(f'input: {vector_to_sentence(input_data[index],input_lang)}')
    print(f'teacher: {vector_to_sentence(teacher_data[index],target_lang)}')
    print(f'target: {vector_to_sentence(target_data[index].flatten(),target_lang)}')
    print(f'predict: {vector_to_sentence(np.argmax(model.predict([[input_data[index]],[teacher_data[index]]]),axis=-1)[0],target_lang)}')

testf(0)

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        elif self.path.startswith("/request?"):
          line=self.path.split('?')[-1]
          line=urllib.parse.unquote(line)
          if len(line)>0:
            print('input: ',line)
            try:
              predict=translate(line.lower(), encoder_model, inf_model)[1:]
            except:
              predict=''
          else:
            predict=''
          print('predicted: ',predict)
          self.send_response(200)
          self.send_header("Content-type", "application/json")
          self.end_headers()
          self.wfile.write(json.dumps({'complete':predict}).encode('utf-8'))
        else:
          return http.server.SimpleHTTPRequestHandler.do_GET(self)
 
def test(HandlerClass = SimpleHTTPRequestHandler,
         ServerClass = http.server.HTTPServer):
    http.server.test(HandlerClass, ServerClass, port=8000)
 
if __name__ == '__main__':
    test()