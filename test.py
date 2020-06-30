import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
import data_utils
import pickle


MAX_LEN = 128
batch_size = 32
model = torch.load("bert_hidden.pt")
print(model)
"""
# Get the GPU device -----------------------------------------------------------
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# prepare dataset -----------------------------------------------------------------
df = pd.read_json('./test.json')
sentences = df.iloc[:, 6].values
labels = df.iloc[:, 5].values.tolist()
for i in range(len(labels)):
    if(labels[i] == "Recommended"):
        labels[i] = 1
    else:
        labels[i] = 0

# tokenizer -----------------------------------------------------------------------
dataset = data_utils.build_data(sentences, MAX_LEN, labels)
dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = batch_size)

def confusion_matrix(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    TP = 0
    TN = 0
    predict_true = 0
    labels_true = 0
    for i in range(pred_flat.shape[0]):
        if(pred_flat[i] == 1 and labels_flat[i] == 1):
            TP += 1
        elif(pred_flat[i] == 0 and labels_flat[i] == 0):
            TN += 1
        if(pred_flat[i] == 1):
            predict_true += 1
        if(labels_flat[i] == 1):
            labels_true += 1
    return TP, TN, predict_true, labels_true

TP = 0
TN = 0
predict_true = 0
labels_true = 0
for batch in dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    with torch.no_grad():        
        loss, logits, hidden_states = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
        
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    t1, t2, t3, t4 = confusion_matrix(logits, label_ids)
    TP += t1
    TN += t2
    predict_true += t3
    labels_true += t4
        
# Report the final accuracy for this validation run.
accuracy = (TP + TN) / len(labels)
precision = TP / predict_true
recall = TP / labels_true

print("Bert for classification:")
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1 score:", 2*precision*recall / (precision+recall))
"""