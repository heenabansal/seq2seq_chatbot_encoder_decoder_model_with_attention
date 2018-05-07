
# coding: utf-8

# In[13]:


from keras.layers import Input, Embedding, LSTM, GRU,Dense, Masking, RepeatVector, Dropout, merge,TimeDistributed, Flatten, Permute, Lambda
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate, dot
from text import Tokenizer
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.utils import plot_model,multi_gpu_model
from sklearn.model_selection import train_test_split


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
import os
multi_gpu_use = False
trial = True
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="2"
if(multi_gpu_use):
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


# In[3]:


if(trial):
    dictionary_size = 100   ##index 0 is not given to any word
    word_embedding_size = 50
    adam_learning_rate = 0.0001
    sentence_embedding_size = 64
    weights_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    GLOVE_DIR = '../glove_pretrained_embeddings/'#'../glove.6B/'
    n_test = 10
    glove_file_name = 'glove.6B.50d.txt'
    n_Epochs = 5
    BatchSize = 128
else:
    dictionary_size = 10000   ##index 0 is not given to any word
    word_embedding_size = 100
    adam_learning_rate = 0.0001
    sentence_embedding_size = 128
    weights_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    GLOVE_DIR = '../glove_pretrained_embeddings/'
    n_test = 10000
    glove_file_name = 'glove.6B.100d.txt'
    n_Epochs = 100
    BatchSize = 512
if(multi_gpu_use):
    BatchSize = 1024


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
    return questions, answers


# In[5]:


def add_start_and_end_symbol(questions, answers):
    questions_lines = questions.split('\n')
    answers_lines = answers.split('\n')
    questions_lines = ['BOS '+p+' EOS' for p in questions_lines]
    answers_lines = ['BOS '+p+' EOS' for p in answers_lines]
    return questions_lines, answers_lines


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


questions, answers = read_file('movie_lines.txt')
questions_lines, answers_lines = add_start_and_end_symbol(questions,answers)
questions_lines_inds = [j for j,item in enumerate(questions_lines) if(len(item.split())>50)]
ans_lines_inds = [j for j,item in enumerate(answers_lines) if(len(item.split())>50)]
inds = list(set(range(len(questions_lines))) - set(questions_lines_inds+ans_lines_inds))
questions_lines = [questions_lines[i] for i in inds]
answers_lines = [answers_lines[i] for i in inds]
total_lines = questions_lines + answers_lines
tokenizer1, encoded_questions, encoded_answers, encoded_total = fit_sequences(questions_lines, answers_lines, 
                                                                              total_lines,nwords=dictionary_size,
                                                                              oov_symbol='oov')
Q, A, maxlen_input, maxlen_output = pad_everything(encoded_questions, encoded_answers)


# In[9]:


vocabulary = tokenizer1.word_index
print tokenizer1.word_index['oov']
print Q.shape, A.shape, maxlen_input, maxlen_output


# In[10]:


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, glove_file_name))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[11]:


embedding_matrix = np.zeros((dictionary_size, word_embedding_size))
# Using the Glove embedding:
for word in vocabulary:
    embedding_vector = embeddings_index.get(word)
    index = vocabulary[word]
    if (embedding_vector is not None) and (index < dictionary_size):
        embedding_matrix[index] = embedding_vector

del embeddings_index
import gc
gc.collect()


# In[14]:


#https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
#http://nmt-keras.readthedocs.io/en/latest/tutorial.html#nmt-model-tutorial (keras nmt module)
n_samples = Q.shape[0]
ad = Adam(lr=adam_learning_rate) 
encoder_inputs = Input(shape=(None,), dtype='int32', name='Encoderinput')
shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix],
                             mask_zero=True,input_length=None, name='Shared_Embedding')

embedded_encoder_inputs = shared_Embedding(encoder_inputs)
encoder_lstm = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', 
            return_sequences=True,return_state=True,name='Encoder_lstm')
encoder_outputs, encoder_h, encoder_c = encoder_lstm(embedded_encoder_inputs)

decoder_inputs = Input(shape=(None,), name="DecoderInput_1")
embedded_decoder_inputs = shared_Embedding(decoder_inputs)
decoder_lstm = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', return_sequences=True,
               name='Decoder_lstm',return_state=True)
decoder_outputs,decoder_h, decoder_c = decoder_lstm(embedded_decoder_inputs, initial_state=[encoder_h,encoder_c])
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

# Has another weight + tanh layer as described in equation (5) of the paper
decoder_dense1 = TimeDistributed(Dense(sentence_embedding_size/2,activation="tanh"))
decoder_dense2 = TimeDistributed(Dense(dictionary_size,activation="softmax"))
output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
output = decoder_dense2(output) # equation (6) of the paper

model = Model([encoder_inputs, decoder_inputs], output)
if(multi_gpu_use):
    model = multi_gpu_model(model, gpus=2)

model.compile(loss='categorical_crossentropy', optimizer=ad)

#plot_model(model,'nmt_encoder_decoder_with_attention.png',show_shapes=True)


# ## Inference Model

# In[15]:


encoder_model = Model(encoder_inputs, [encoder_outputs,encoder_h,encoder_c])
#plot_model(encoder_model,'nmt_encoder_with_attention.png',show_shapes=True)

decoder_state_input_h = Input(shape=(sentence_embedding_size,), name="DecoderStateInput_1")
decoder_state_input_c = Input(shape=(sentence_embedding_size,), name="DecoderStateInput_2")
encoder_outputs = Input(shape=(maxlen_input,sentence_embedding_size), name="Encoder_outputs")

decoder_inputs = Input(shape=(None,), name="DecoderInput_1")
embedded_decoder_inputs = shared_Embedding(decoder_inputs)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(embedded_decoder_inputs, 
                                                initial_state=[decoder_state_input_h,decoder_state_input_c])
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

output = decoder_dense1(decoder_combined_context) # equation (5) of the paper
output = decoder_dense2(output) # equation (6) of the paper

decoder_model = Model(
    [decoder_inputs] + [decoder_state_input_h,decoder_state_input_c] + [encoder_outputs],
    [output] + [decoder_state_h,decoder_state_c])
#plot_model(decoder_model,'nmt_decoder_with_attention.png',show_shapes=True)


# In[16]:


def find_Y(ans):
    y = [np.append(line[1:],[0]) for line in ans]
    y = to_categorical(np.asarray(y),dictionary_size)
    return y


# In[17]:


if(trial):
    Qtest = Q[0:n_test,:]
    Atest = A[0:n_test,:]
    Qtrain = Q[n_test + 1:n_test + 200,:]
    Atrain = A[n_test + 1:n_test + 200,:]
    #Ytrain = find_Y(Atrain)
else:
    Qtest = Q[0:n_test,:]
    Atest = A[0:n_test,:]
    Qtrain = Q[n_test + 1:,:]
    Atrain = A[n_test + 1:,:]
    #Ytrain = find_Y(Atrain)

Qtrain, Qval, Atrain, Aval = train_test_split(Qtrain, Atrain, test_size=0.2, random_state=123)


# In[18]:


def generate_data(Qtrain,Atrain,batch_size):
    while True:
        for i in range(0,len(Atrain),batch_size):
            Atrain_sample = Atrain[i:i+batch_size,:]
            Qtrain_sample = Qtrain[i:i+batch_size,:]
            Ytrain_sample = find_Y(Atrain_sample)
            yield [Qtrain_sample,Atrain_sample], Ytrain_sample


# In[19]:


checkpoint_callback = ModelCheckpoint(weights_file, monitor='val_loss', verbose=0, 
                           save_best_only=True, mode='auto', period=5)
tensorboard_callback = TensorBoard(log_dir='./logs', batch_size=BatchSize)

steps = int(len(Qtrain)/BatchSize)+1
Yval = find_Y(Aval)
model.fit_generator(generate_data(Qtrain,Atrain,BatchSize),steps_per_epoch=steps,use_multiprocessing=False,
                    epochs=n_Epochs,validation_data=([Qval,Aval],Yval),
                    callbacks=[checkpoint_callback,tensorboard_callback])
#model.fit([Qtrain, Atrain], Ytrain, batch_size=BatchSize, epochs=1,callbacks=[callback,tensorboard_callback],
          #validation_split=0.1)


# In[ ]:


def beam_search_decoder(input_seq,max_sequences):
    encoder_outputs, h, c = encoder_model.predict(input_seq)
    sequences = [[[vocabulary['bos']],1.0, h ,c]]
    target_seq = np.zeros((1, 1))
    target_seq[0,0] = vocabulary['bos']
    stop_condition = False
    decoded_sentence = ''
    last_level_words = [vocabulary['bos']]
    for i in range(maxlen_input):
        #generate candidates for all the last level words
        all_candidates = []
        for j in range(len(sequences)):
            seq, score, h, c = sequences[j]
            last_word = seq[-1]
            target_seq[0,0] = last_word
            output, h, c = decoder_model.predict([target_seq] + [h,c] + [encoder_outputs])
            output = output[0,0]
            for k in range(len(output)):
                candidate = [seq + [k], score * -np.log(output[k]), h, c]
            all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:max_sequences]
    decoded_sentence = sequences[0][0]
    decoded_sentence = [tokenizer1.index_word[x] for x in decoded_sentence if(x!=0)]
    decoded_sentence = ' '.join(decoded_sentence)
    return decoded_sentence


# In[ ]:


def greedy_decoder(input_seq):
    encoder_outputs, h, c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0,0] = vocabulary['bos']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output, h, c = decoder_model.predict([target_seq] + [h,c] + [encoder_outputs])
        sampled_token_index = np.argmax(output[0, -1, :])
        if(sampled_token_index!=0):
            sampled_word = tokenizer1.index_word[sampled_token_index]
            decoded_sentence += " "+sampled_word

        if (sampled_word == 'eos' or len(decoded_sentence) > maxlen_input):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq[0, 0] = sampled_token_index
    return decoded_sentence


# In[ ]:


for i in range(n_test):
    question = [tokenizer1.index_word[x] for x in Qtest[i] if(x!=0)]
    question = ' '.join(question)
    ans = [tokenizer1.index_word[x] for x in Atest[i] if(x!=0)]
    ans = ' '.join(ans)
    print('Question:'+question)
    print('Answer:'+ans)
    ques = np.reshape(Qtest[i],(1,len(Qtest[i])))
    #print('Predicted:'+greedy_decoder(ques))
    print('Predicted:'+beam_search_decoder(ques,20))

