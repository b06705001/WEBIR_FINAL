import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.feature_extraction.text import TfidfVectorizer

k_neighbor = 5
if(k_neighbor % 2 == 0):
    print("error: k neigbors has to be odd")

class knn(nn.Module):
    def __init__(self, train_vec):
        super(knn, self).__init__()
        self.train_vec = train_vec

    def forward(self, test_vec):
        test_vec = test_vec.permute(1, 0)
        prediction = torch.matmul(self.train_vec, test_vec)
        prediction = prediction.detach().cpu().numpy()
        return prediction

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
		label.append(1)
	else:
		label.append(0)
label = np.array(label)
for i in t_l:
	if (i=="Recommended"):
		t_label.append(1)
	else:
		t_label.append(0)

dataset = TensorDataset(sentences_vec_test)
dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 2048)
model = knn(sentences_vec)
ans = []
cnt = 0
for batch in dataloader:
    test_vec = batch[0]
    matrix = model.forward(test_vec)
    matrix = matrix.transpose()
    for i in range(len(test_vec)):
        k_max = matrix[i].argsort()[-k_neighbor:][::-1]
        temp = 0
        for idx in k_max:
            temp += label[idx]
        ans.append(temp)
    cnt += 1
    print("KNN:", cnt)

for i in range(len(ans)):
    if(ans[i] >= (k_neighbor//2 + 1)):
        ans[i] = 1
    else:
        ans[i] = 0


TP, TN, predict_true, labels_true = confusion_matrix(ans, t_label)
accuracy = (TP + TN) / len(t_label)
precision = TP / predict_true
recall = TP / labels_true
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1 score:", 2*precision*recall / (precision+recall))
