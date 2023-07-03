# !pip install conll-df
# !pip install keras_preprocessing
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import pandas as pd
from conll_df import conll_df
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
# from gensim.models.keyedvectors import KeyedVectors
# from gensim.models import FastText
from keras.callbacks import EarlyStopping
# from gensim.models import KeyedVectors
from keras.utils import to_categorical
import urllib.request, zipfile
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import re
import pickle
from sklearn.utils import class_weight
import copy

# path = '/kaggle/input/datasetnlp/en_atis-ud-train.conllu'
# df = conll_df(path, file_index=False)
# df.head(40)

def pos_tag(path):
    df = conll_df(path, file_index=False)
    inde=[]
    for x in df.index:
        if x[0] not in inde:
            inde.append(x[0])
    word_pos_pairs=[]
    for index in inde:
        z=list(zip(df.loc[index]["l"].tolist(), df.loc[index]["x"].tolist()))
        word_pos_pairs.append(z)
    return word_pos_pairs

# train_sentences = pos_tag('/kaggle/input/datasetnlp/en_atis-ud-train.conllu')
# val_sentences = pos_tag('/kaggle/input/datasetnlp/en_atis-ud-dev.conllu')
# test_sentences = pos_tag('/kaggle/input/datasetnlp/en_atis-ud-test.conllu')

def prepare_data(tagged_sentences):
    X = [] # store input sequence
    Y = [] # store output sequence
    
    for sentence in tagged_sentences:
        flag=True
        X_sentence = []
        Y_sentence = []
        for word in sentence:         
            X_sentence.append(word[0])  # word[0] contains the word
            Y_sentence.append(word[1])  # word[1] contains corresponding tag
            if(word[1]=="SYM"):
                flag=False
        if flag:        
            X.append(X_sentence)
            Y.append(Y_sentence)
    return X,Y

# x_train,y_train=prepare_data(train_sentences)
# x_test,y_test=prepare_data(test_sentences)
# x_valid,y_valid=prepare_data(val_sentences)

# x_combine=x_train+x_valid+x_test
# y_combine=y_train+y_valid+y_test

# word2index=dict()
# word2index['-pad-']=0
# word2index['unk']=1
# index2word=dict()
# index2word[0]='-pad-'
# index2word[1]='unk'
# tag2index=dict()
# tag2index['-pad-']=0
# tag2index['unk']=1

# index2tag=dict()
# index2tag[0]='-pad-'
# index2tag[1]='unk'

# unique_words=list(set([word.lower() for sentence in x_combine for word in sentence]))
# unique_tags=list(set([word.lower() for sentence in y_combine for word in sentence]))

# word2vec= KeyedVectors.load_word2vec_format('/kaggle/input/glovefiles/glove.6B.300d.txt', binary=False,no_header=True)
# with open('/kaggle/input/w2vfile/w2v (1).pickle', 'rb') as file:
#     word2vec = pickle.load(file)


# words_ex = word2vec.index_to_key[:50000]

# for x in words_ex:
#     if x not in unique_words:
#         unique_words.append(x)

# def prepare_token(dic1,dic2,words):
#     idx=2
#     for word in words:
#         dic1[word.lower()]=idx
#         dic2[idx]=word
#         idx+=1
#     return dic1,dic2


# word2index,index2word=prepare_token(word2index,index2word,unique_words)
# tag2index,index2tag=prepare_token(tag2index,index2tag,unique_tags)

with open('word2index_f.pickle', 'rb') as file:
    word2index = pickle.load(file)
# with open('index2word_f.pickle', 'rb') as file:
#     index2word = pickle.load(file)
with open('tag2index_f.pickle', 'rb') as file:
    tag2index = pickle.load(file)
# with open('index2tag_f.pickle', 'rb') as file:
#     index2tag = pickle.load(file)





def fetch_token(dataset,dict1):
    new_dataset=copy.deepcopy(dataset)
    i=0
    j=0
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            try:
                new_dataset[i][j]=dict1[dataset[i][j].lower()]
            except:
                new_dataset[i][j]=dict1['unk']
    return new_dataset



num_words = len(word2index)
# print("Vocabulary volume: {}".format(num_words))

num_tags   = len(tag2index)
# num_tags=15
# print("Total number of tags =: {}".format(num_tags))

# lengths = [len(seq) for seq in x_combine]
# max_len=max(lengths)
max_len=46
# print("Length of longest sentence is: {}".format(max_len))
labels_name=([sentence.lower() for sentence in tag2index.keys()])
# labels_name=['-pad-', 'unk', 'propn', 'verb', 'det', 'num', 'adv', 'pron', 'cconj', 'noun', 'aux', 'adj', 'adp', 'intj', 'part']
# print(labels_name)
def do_padding(tokenized,mx_len):
    padded_data=tf.keras.preprocessing.sequence.pad_sequences(tokenized,maxlen=mx_len,padding="post")
    return padded_data

# y_test_token=fetch_token(y_test,tag2index)


def pre_processing(mx_len,dict1,dataset):
    toks=fetch_token(dataset,dict1)
#     toks = np.array(toks)
#     return pad_sequences(toks ,maxlen=46,  padding="post")
    padded=do_padding(toks,mx_len)
#         ret_data.append(padded)
    return padded

# X_train=pre_processing(max_len,word2index,x_train)
# X_valid=pre_processing(max_len,word2index,x_valid)
# X_test=pre_processing(max_len,word2index,x_test)
# Y_train=pre_processing(max_len,tag2index,y_train)
# Y_valid=pre_processing(max_len,tag2index,y_valid)
# Y_test=pre_processing(max_len,tag2index,y_test)
# Y_tr_cat = to_categorical(Y_train)
# Y_val_cat = to_categorical(Y_valid)
# Y_test_cat = to_categorical(Y_test)

def create_embeddings(embed_size,voc_size,tokenizer,word2vec):
    # create an empty embedding matix
#     print(type(word2vec))
    embedding_weights = np.zeros((voc_size, embed_size))
    # create a word to index dictionary mapping
    word2id = tokenizer
    for word, index in word2id.items():
        try:
            embedding_weights[index, :] =  word2vec.get_vector(word)[:embed_size]
        except KeyError:
            pass
#     zeros = np.all(embedding_weights == 0, axis=1)

# # Count the number of rows that are all zeros
#     count = np.count_nonzero(zeros)

#     # Print the count
#     print("Number of rows that are all zeros:", count)
    return embedding_weights

def create_model(vocc_size,mx_len,embed_size,embedding_weights,num_layers,dropout,tags_len,learning_rate,hidden_units):
    model = Sequential()
    model.add(Embedding(input_dim=vocc_size, output_dim=embed_size,input_length=46, weights=[embedding_weights], trainable=False,mask_zero=True))
    for i in range(num_layers):
        model.add(LSTM(hidden_units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add((Dense(tags_len, activation='softmax')))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model

def train_model(model,X_train,Y_tr_cat,X_valid,Y_val_cat,model_name):
    my_callbacks = [EarlyStopping(monitor="val_acc", patience=3, verbose=1)]
    history=model.fit(X_train, Y_tr_cat, epochs=20, verbose=1,validation_data=(X_valid, Y_val_cat) ,callbacks=my_callbacks)
    model.save(model_name+".h5")
    return model,history

def pred_index(pred_labels):
    return np.argmax(pred_labels,axis=2)

def model_evaluate(model,y_test,y_pred_ind,labels_name):
    y_pred_final = []
    y_true_final = []
    for y_pred_i, y_true_i in zip(y_pred_ind, y_test):
        for x in y_pred_i[:len(y_true_i)]:
            y_pred_final.append(x)
        for e in y_true_i:
            y_true_final.append(e)

    # print(classification_report(y_true_final, y_pred_final,target_names=labels_name))
class_label=[]
for i in range(2,len(labels_name)):
    class_label.append(labels_name[i])

def plot_graph(model):
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()

def run_fun(word2vec):
    # Load the pre-trained model from the file path
    voc_size=len(word2index)
    
    embed_sizes = [300, 128]
    learning_rates = [0.001, 0.0001]
    mx_len = 46
    num_layers_list = [1, 2]
    dropouts = [0.2, 0.8]
    hidden_units = [128, 64]
    model_name="model_"
    ids=1
    for embed_size in embed_sizes:
        embedding=create_embeddings(embed_size,voc_size,word2index,word2vec)
        for dropout in dropouts:
            for hidden_unit in hidden_units:
                for learning_rate in learning_rates:
                    for num_layers in num_layers_list:
                        model=create_model(voc_size,mx_len,embed_size,embedding,num_layers,dropout,tags_len,learning_rate,hidden_unit)
                        model,history=train_model(model,X_train,Y_tr_cat,X_valid,Y_val_cat,model_name+str(ids))
                        ids+=1
                        plot_graph(history)
                        y_pred=model.predict(X_test)
                        y_pred_ind=pred_index(y_pred)
                        model_evaluate(model,y_test_token,y_pred_ind,class_label)

# run_fun(word2vec)

def handle_preprocess_sent(strings):
    strings=strings.strip()
    strings=re.sub(' +',' ',strings)
    strings=strings.split()
    return strings

def pre_processing1(tok_seq,dataset):
#     print(dataset)
    toks=fetch_token(dataset,tok_seq)
#     print(toks)
    toks = np.array(toks)
    padded=do_padding(toks,46)
#     print(padded)
    return padded

def invoke_fun(path_to_model,tags,word2index):
    model = load_model(path_to_model)
    ip = input("input sentence: ")
    token_sen=handle_preprocess_sent(ip)
    token1=[]
    for e in token_sen:
        token1.append(e)
#     token1=token_sen
    # print(token1)
    for idx in range(len(token_sen)):
        token_sen[idx]=token_sen[idx].lower()
    X_test1=[]
    X_test1.append(token_sen)
    padded=pre_processing1(word2index,X_test1)
#     print(padded.shape)
    y_pred=model.predict(padded)
    y_pred_ind=pred_index(y_pred)
    y_pred_final = []
    idss=0
    # print(token1)
    for y_pred_i in y_pred_ind:
#         print(y_pred_i)
        for x in y_pred_i[:len(token_sen)]:
            print(str(token1[idss])+"\t"+tags[x].upper())
            idss+=1

invoke_fun("model_1.h5",labels_name,word2index)