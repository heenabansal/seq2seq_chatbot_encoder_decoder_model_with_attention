
# coding: utf-8

# In[1]:


from keras.layers import Input, Embedding, LSTM, Dense, Masking, RepeatVector, Dropout, merge,TimeDistributed, Flatten, Permute, Lambda
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate
from text import Tokenizer
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
# In[2]:


import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import os.path
import sys
import nltk
import re
import time
from keras.utils import plot_model
import pandas as pd


# In[3]:


dictionary_size = 500   ##index 0 is not given to any word
weights_file = 'my_model_weights.h5'


# In[4]:


def read_file(filename):
    f1 = open(filename,'r')
    data = f1.readlines()
    f1.close()
    questions = []
    answers = []
    for i,line in enumerate(data):
        line = line.split(' +++$+++ ')
        if(i%2==0):
            questions.append(line[4])
        else:
            answers.append(line[4])
    questions.pop()
    print("number of questions={},answers={}".format(len(questions),len(answers)))
    print questions[0], answers[0]
    questions = ' '.join(questions)
    answers = ' '.join(answers)
    total = questions + answers
    return questions, answers, total


# In[5]:


def add_start_and_end_symbol(questions, answers, total):
    questions_lines = questions.split('\n')
    answers_lines = answers.split('\n')
    total_lines = total.split('\n')
    questions_lines = ['BOS '+p+' EOS' for p in questions_lines]
    answers_lines = ['BOS '+p+' EOS' for p in answers_lines]
    total_lines = ['BOS '+p+' EOS' for p in total_lines]
    return questions_lines, answers_lines, total_lines


# In[6]:


def fit_sequences(questions_lines, answer_lines, total_lines, nwords=dictionary_size, oov_symbol='oov'):
    # create the tokenizer
    t = Tokenizer(num_words=dictionary_size,oov_token=oov_symbol)
    # fit the tokenizer on the documents
    t.fit_on_texts(total_lines)
    # summarize what was learned
    #print(t.word_counts)
    #print(t.document_count)
    #print(t.word_index)
    #print(t.word_docs)
    # integer encode documents
    encoded_questions = t.texts_to_sequences(questions_lines)
    encoded_answers = t.texts_to_sequences(answer_lines)
    encoded_total = t.texts_to_sequences(total_lines)
    return t, encoded_questions, encoded_answers, encoded_total


# In[7]:


def pad_everything(encoded_questions, encoded_answers):
    maxlen_input = max([len(x) for x in encoded_questions])
    maxlen_output = max([len(x) for x in encoded_answers])
    Q = sequence.pad_sequences(encoded_questions, maxlen=maxlen_input)
    A = sequence.pad_sequences(encoded_answers, maxlen=maxlen_output, padding='post')
    return Q, A, maxlen_input, maxlen_output


# In[8]:


questions, answers, total = read_file('movie_lines.txt')
questions_lines, answers_lines, total_lines = add_start_and_end_symbol(questions,answers, total)
tokenizer1, encoded_questions, encoded_answers, encoded_total = fit_sequences(questions_lines, answers_lines, 
                                                                              total_lines,nwords=dictionary_size,oov_symbol='oov')
Q, A, maxlen_input, maxlen_output = pad_everything(encoded_questions, encoded_answers)


# In[9]:


vocabulary = tokenizer1.word_index
tokenizer1.word_index['oov']


# In[10]:


print Q.shape, A.shape, maxlen_input, maxlen_output


# # Train Model

# In[11]:


from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence


# In[12]:


word_embedding_size = 100
sentence_embedding_size = 256
weights_file = 'my_model_weights20.h5'
GLOVE_DIR = '../glove.6B/'
n_test = 10000
n_samples = Q.shape[0]
n_subsets = 1
n_Epochs = 100


# In[13]:


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[14]:


embedding_matrix = np.zeros((dictionary_size, word_embedding_size))
# Using the Glove embedding:
for word in vocabulary:
    embedding_vector = embeddings_index.get(word)
    index = vocabulary[word]
    if (embedding_vector is not None) and (index < dictionary_size):
        embedding_matrix[index] = embedding_vector


# In[15]:


del embeddings_index
import gc
gc.collect()


# In[16]:


# *******************************************************************
# Keras model of the chatbot: 
# *******************************************************************

ad = Adam(lr=0.001) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='input_context')
input_answer = Input(shape=(maxlen_input,), dtype='int32', name='input_answer')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform')
shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix],
                             mask_zero=True,input_length=maxlen_input)
word_embedding_context = shared_Embedding(input_context)
mask_layer = Masking(mask_value=0)
context_embedding = LSTM_encoder(mask_layer(word_embedding_context))

word_embedding_answer = shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(mask_layer(word_embedding_answer))

merge_layer = Concatenate(axis=1)([context_embedding, answer_embedding])
out = Dense(dictionary_size/2, activation="relu")(merge_layer)
out = Dense(dictionary_size, activation="softmax")(out)

model = Model(inputs=[input_context, input_answer], outputs = [out])

model = multi_gpu_model(model, gpus=2)

model.compile(loss='categorical_crossentropy', optimizer=ad)


# In[17]:


plot_model(model,to_file='model.png',show_shapes=True)


# In[25]:


def greedy_decoder(question):
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocabulary['bos']  #put bos in the end of ans_partial (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([question, ans_partial])
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]   #shift ans_partial one place left 
        ans_partial[0, -1] = mp            #replace the last character of the ans_partial with generated one
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if (k < dictionary_size) and (k>0):    #index 0 is not assigned to any word
            w = tokenizer1.index_word[k]     #word to index dictionary
            text = text + w + ' '
    return(text)


def beam_search_decoder(question):
    max_sequences = 10
    start = vocabulary['bos']
    sequences = [[[start], 1.0]]   
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocabulary['bos']  #put bos in the end of ans_partial (begin of sentence)
    for k in range(maxlen_input - 1):
        row = model.predict([question, ans_partial])
        row=row[0]
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):      
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:max_sequences]
        #update ans_partial to calculate next row in next iteration
        seq, score = sequences[0]
        ans_partial[0,-(k+2):] = seq
    final_sequence = sequences[0][0]
    return sequences[0][0]


# ### divide data into train test

# In[21]:


Qtest = Q[0:n_test,:]
Atest = A[0:n_test,:]
Qtrain = Q[n_test + 1:,:]
Atrain = A[n_test + 1:,:]


# ### convert training data answers into trailings and replicate questions for the same

# In[22]:


def roll_answers(Qtrain, Atrain):
    Qtrain_extended = []
    Atrain_extended = []
    Ytrain = []
    for i,sample in enumerate(Atrain):
        for k,item in enumerate(sample):
            if(item ==vocabulary['eos']):
                break
            Qtrain_extended.append(Qtrain[i])    #replicating Q_train
            Atrain_partial = np.zeros(maxlen_input)
            Atrain_partial[-(k+1):] = Atrain[i,0:(k+1)]   #push words from 0 to k in sentence i
            Atrain_extended.append(Atrain_partial)
            Ytrain.append(Atrain[i,(k+1)])         #next word in sentence i
    Qtrain_extended = np.asarray(Qtrain_extended)
    Atrain_extended = np.asarray(Atrain_extended)
    Ytrain = to_categorical(np.asarray(Ytrain),dictionary_size)
    return Qtrain_extended, Atrain_extended, Ytrain


# In[23]:


nsamples = len(Atrain)
batch_size = 100
BatchSize = 1024


# In[ ]:


for m in range(n_Epochs):
    for i in range(0,nsamples,batch_size):
        Qtrain_extended, Atrain_extended, Ytrain = roll_answers(Qtrain[i:i+batch_size,:], Atrain[i:i+batch_size,:])
        print Qtrain_extended.shape, Atrain_extended.shape, Ytrain.shape
        print('Training epoch: %d, training examples: %d - %d'%(m,i,i+batch_size))
        model.fit([Qtrain_extended, Atrain_extended], Ytrain, batch_size=BatchSize, epochs=1)
        
    test_input = Qtest[41:42]
    print(greedy_decoder(test_input))
    train_input = Qtrain[1:2]
    print(greedy_decoder(train_input))   
    model.save_weights(weights_file, overwrite=True)


# In[ ]:


#https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html

