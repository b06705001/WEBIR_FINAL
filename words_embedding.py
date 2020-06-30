import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
import data_utils
import pickle
import sys

MAX_LEN = 128
batch_size = 32
model = torch.load("bert_hidden.pt")
input_file = "./train.json"
token_store_name = "sentences_token_train"
embedding_store_name = "sentences_embedding_train"
isTest = sys.argv[1]
print(isTest)
if(isTest == 'true'):
    input_file = './test.json'
    token_store_name = "sentences_token_test"
    embedding_store_name = "sentences_embedding_test"

def isword(word):
    stop = ['.', ',', '(', ')', '/', ';', ':', '\'', '"', '?', '!', '<', '>', '@', 
            '#', '$', '%', '^', '&', '&', '*', '-', '+', '=', '_', '[', ']', '{', '}']
    if(word == "[UNK]"):
        return 0
    elif(len(word) > 1):
        if(word[0] == '#' and word[1] == '#'):
            return 2
        else:
            return 1
    elif(word not in stop):
        return 1
    else:
        return 0

    

# Get the GPU device -----------------------------------------------------------
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# prepare dataset -----------------------------------------------------------------
df = pd.read_json(input_file)
sen_temp = df.iloc[:, 6].values
size = 120000
looptimes = len(sen_temp) // size + 1

for loop in range(looptimes):
    start = loop*size
    end = min((loop+1)*size, len(sen_temp))
    sentences = sen_temp[start:end]

    # tokenizer -----------------------------------------------------------------------
    dataset = data_utils.build_data(sentences, MAX_LEN)
    dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = batch_size)

    # (1) which sentence
    # (2) which words
    sentences_token = []
    # (1) which sentence
    # (2) which words
    # (3) embedding
    sentences_embedding = []

    # split sentence into tokens.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    split = []
    for i in range(len(sentences)):
        split.append(tokenizer.tokenize(sentences[i]))


    # forwarding
    sen_cnt = 0
    for batch in dataloader:
        if(sen_cnt % 10000 == 0):
            print(sen_cnt)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad(): 
            logits, hidden_states = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # Concatenate the tensors for all layers.
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = token_embeddings.permute(1, 2, 0, 3)

            for i in range(len(b_input_ids)):
                add_cnt = 0
                word_cnt = -1
                isFirst = True
                sum_vec = torch.zeros((768, )).to(device)
                word_vec = []
                temp = []
                for j in range( min(len(split[sen_cnt]), 127) ):
                    word = split[sen_cnt][j]
                    if(isword(word) == 1):
                        temp.append(word)
                        word_cnt += 1
                        if(isFirst is not True):
                            # 把前一個加進去
                            word_vec.append(sum_vec.detach().cpu().numpy() / add_cnt)
                            add_cnt = 0
                        isFirst = False
                        sum_vec = torch.zeros((768, )).to(device)
                        sum_vec += torch.sum(token_embeddings[i, j+1, 1:], dim=0)  # j+1, 因為第一個是[CLT]
                    elif(isword(word) == 2):
                        temp[word_cnt] = temp[word_cnt][:-2] + word[2:]
                        sum_vec += torch.sum(token_embeddings[i, j+1, 1:], dim=0)
                    add_cnt += 1

                sen_cnt += 1
                if(isFirst is not True):
                    word_vec.append(sum_vec.detach().cpu().numpy() / add_cnt)
                temp = np.array(temp)
                word_vec = np.array(word_vec)
                sentences_token.append(temp)
                sentences_embedding.append(word_vec)

    sentences_token = np.array(sentences_token)
    np.save(token_store_name + str(loop) + '.npy', sentences_token)
    del sentences_token
    del sentences
    del dataloader
    del dataset
    del split
    sentences_embedding = np.array(sentences_embedding)
    np.save(embedding_store_name + str(loop) + '.npy', sentences_embedding)
