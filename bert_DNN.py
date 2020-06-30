import numpy as np
import json
import pandas as pd
import torch

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def confusion_matrix(preds, labels):
    TP = 0
    TN = 0
    predict_true = 0
    labels_true = 0
    for i in range(len(preds)):
        if(preds[i] == 1 and labels[i] == 1):
            TP += 1
        elif(preds[i] == 0 and labels[i] == 0):
            TN += 1
        if(preds[i] == 1):
            predict_true += 1
        if(labels[i] == 1):
            labels_true += 1
    return TP, TN, predict_true, labels_true

sentences_vec = np.load("sentences_vec_train.npy")
sentences_vec_test = np.load("sentences_vec_test.npy")
sentences_vec = torch.from_numpy(sentences_vec)
sentences_vec_test = torch.from_numpy(sentences_vec_test)

df = pd.read_json('./train.json')
l = df.iloc[:, 5].values
df = pd.read_json('./test.json')
t_l = df.iloc[:, 5].values

label = []
t_label = []
for i in l:
	if (i=="Recommended"):
		label.append([0, 1])
	else:
		label.append([1, 0])
label = np.array(label)
for i in t_l:
	if (i=="Recommended"):
		t_label.append(1)
	else:
		t_label.append(0)

adam=Adam(lr=0.001)
model = Sequential()
model.add(Dense(2, activation='softmax', input_dim=sentences_vec.shape[1]))
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(sentences_vec, label, epochs=20, batch_size=512, validation_split=0)
model.save_weights("model1.h5")
ans=[]
p1=model.predict(sentences_vec_test)
for i in range(len(p1)):
	if(p1[i][0]>=0.5):
		ans.append(0)
	else:
		ans.append(1)

TP, TN, predict_true, labels_true = confusion_matrix(ans, t_label)
accuracy = (TP + TN) / len(t_label)
precision = TP / predict_true
recall = TP / labels_true
print("bert with TF-IDF and DNN")
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1 score:", 2*precision*recall / (precision+recall))
