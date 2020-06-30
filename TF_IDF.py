import pandas as pd
import numpy as np
import json
import os
import pickle
import sys
import time
import sklearn.neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Masking,GRU,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import tensorflow as tf
import csv
from sklearn import preprocessing
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
my_stop_words = ['seemed', 'without', 'anything', 'might', 'side', 'more', 'where', 'eleven', 'yet', 'always', 'mine', 'him', 'who', 'up', 'when', 'what', 'above', 'fire', 'mill', 'over', 'my', 'this', 'would', 'please', 'once', 'before', 'our', 'we', 'therefore', 'be', 'eight', 'do', 'cannot', 'every', 'due', 'become', 'forty', 'serious', 'detail', 'bottom', 'meanwhile', 'show', 'than', 'thereafter', 'ltd', 'somewhere', 'yourself', 'interest', 'already', 'very', 'under', 'amoungst', 'some', 'seems', 'they', 'whereafter', 'hereupon', 'therein', 'but', 'bill', 'describe', 'per', 'amongst', 'except', 'whenever', 'hundred', 'co', 'twelve', 'whether', 'there', 'off', 'must', 'one', 'call', 'thereby', 'seeming', 'whoever', 'or', 'take', 'another', 'anyway', 'being', 'rather', 'so', 'ours', 'between', 'noone', 'cant', 'namely', 'these', 'to', 'here', 'of', 'few', 'put', 'de', 'which', 'else', 'his', 'almost', 'often', 'other', 'eg', 'too', 'could', 'whole', 'can', 'further', 'anyhow', 'thick', 'sixty', 'from', 'everyone', 'them', 'in', 'amount', 'into', 'after', 'those', 'it', 'the', 'myself', 'out', 'at', 'something', 'nor', 'through', 'name', 'whereby', 'beforehand', 'are', 'nine', 'again', 'her', 'down', 'many', 'by', 'third', 'thence', 'becoming', 'a', 'found', 'five', 'however', 'nobody', 'such', 'three', 'latterly', 'keep', 'four', 'whereas', 're', 'have', 'since', 'someone', 'ourselves', 'inc', 'us', 'because', 'behind', 'wherever', 'for', 'below', 'am', 'somehow', 'with', 'each', 'former', 'six', 'still', 'couldnt', 'full', 'twenty', 'whither', 'otherwise', 'you', 'hereby', 'perhaps', 'hence', 'go', 'seem', 'sometime', 'onto', 'your', 'beside', 'me', 'then', 'had', 'their', 'against', 'anyone', 'and', 'himself', 'thru', 'hereafter', 'see', 'ever', 'give', 'was', 'yourselves', 'if', 'nevertheless', 'although', 'on', 'were', 'she', 'themselves', 'around', 'its', 'nowhere', 'part', 'as', 'throughout', 'get', 'towards', 'etc', 'fifty',  'empty', 'neither', 'most', 'via', 'wherein', 'thereupon', 'indeed', 'though', 'why', 'yours', 'moreover', 'becomes', 'un', 'whatever', 'whereupon', 'an', 'whence', 'everything', 'been', 'all', 'elsewhere', 'formerly', 'others', 'find', 'mostly', 'sincere', 'during', 'last', 'move', 'will',  'much', 'while', 'several', 'within', 'until', 'that', 'either', 'ie', 'latter', 'thin', 'may', 'together', 'fifteen', 'any', 'became', 'own', 'done', 'afterwards', 'also', 'he', 'besides', 'both', 'system', 'ten', 'toward', 'hers', 'not', 'two', 'hasnt', 'i', 'alone', 'has', 'only', 'upon', 'whose', 'thus', 'along', 'front', 'itself', 'even', 'is', 'should', 'now', 'among', 'first', 'herself', 'anywhere', 'made', 'next', 'herein', 'about', 'sometimes', 'whom', 'everywhere', 'least', 'fill', 'across', 'how','the',
]
import keras
K = keras.backend
def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 
context=[]
l=[]
t_c=[]
t_l=[]
with open('test.json', 'r') as json_file:
	test = json.loads(json_file.read())
with open('train.json', 'r') as json_file:
	train = json.loads(json_file.read())
for i in range(len(train)):
	context.append(train[i][6])
	l.append(train[i][5])
for i in range(len(test)):
	t_c.append(test[i][6])
	t_l.append(test[i][5])
t_label=[]
label=[]
"""
KNN label
for i in l:
	if (i=="Recommended"):
		label.append(1)
	else:
		label.append(0)

"""
#DNNlabel
for i in l:
	if (i=="Recommended"):
		label.append([0,1])
	else:
		label.append([1,0])
label=np.array(label)
for i in t_l:
	if (i=="Recommended"):
		t_label.append(1)
	else:
		t_label.append(0)
tfidf = TfidfVectorizer(lowercase=True,norm=None,stop_words=my_stop_words,max_features=10000)#轉小寫，沒標準化，default是有標準化
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float64')
    return x 
X = tfidf_features(context, flag="train")
x =	tfidf_features(t_c, flag="test")
print(1)
knn = sklearn.neighbors.KNeighborsClassifier()
#訓練資料集
"""KNN
knn.fit(X, label)
p=knn.predict(x)
print(p)
print(accuracy_score(p, t_label))
"""
"""
0.8073787925821175
0.5423787 ,0.74931492, 0.62927067 none
0.799894227342232
0.54745194,0.72145981,0.62252489stopword
 
0.8078961588429391
0.53471163,0.7565438 , 0.6265728 10000feature

 10000feature DNN
 0.8700146012255832
0.66734056,0.87126494,0.75578884
"""
#DNN
adam=Adam(lr=0.001)
print(X)
print(X.shape)
#X_test = tfidf_features(test_text, flag="test")
model = keras.Sequential()
model.add(keras.layers.Dense(1000, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(X, label, epochs=1, batch_size=512, validation_split=0.1)
model.save_weights("model1.h5")
p=[]
p1=model.predict(x)
for i in range(len(p1)):
	if(p1[i][0]>=0.5):
		p.append(0)
	else:
		p.append(1)

print(accuracy_score(p, t_label))

print(precision_recall_fscore_support(p, t_label))