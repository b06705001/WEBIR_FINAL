import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import sys

isTest = sys.argv[1]
token_input_file =  "sentences_token_train"
embedding_input_file = "sentences_embedding_train"
output_file = "sentences_vec_train.npy"
looptimes = 1
size = 120000
if(isTest == 'true'):
    token_input_file =  "sentences_token_test"
    embedding_input_file = "sentences_embedding_test"
    output_file = "sentences_vec_test.npy"


my_stop_words = ['seemed', 'without', 'anything', 'might', 'side', 'more', 'where', 'eleven', 'yet', 'always', 'mine', 'him', 'who', 'up', 'when', 'what', 'above', 'fire', 'mill', 'over', 'my', 'this', 'would', 'please', 'once', 'before', 'our', 'we', 'therefore', 'be', 'eight', 'do', 'cannot', 'every', 'due', 'become', 'forty', 'serious', 'detail', 'bottom', 'meanwhile', 'show', 'than', 'thereafter', 'ltd', 'somewhere', 'yourself', 'interest', 'already', 'very', 'under', 'amoungst', 'some', 'seems', 'they', 'whereafter', 'hereupon', 'therein', 'but', 'bill', 'describe', 'per', 'amongst', 'except', 'whenever', 'hundred', 'co', 'twelve', 'whether', 'there', 'off', 'must', 'one', 'call', 'thereby', 'seeming', 'whoever', 'or', 'take', 'another', 'anyway', 'being', 'rather', 'so', 'ours', 'between', 'noone', 'cant', 'namely', 'these', 'to', 'here', 'of', 'few', 'put', 'de', 'which', 'else', 'his', 'almost', 'often', 'other', 'eg', 'too', 'could', 'whole', 'can', 'further', 'anyhow', 'thick', 'sixty', 'from', 'everyone', 'them', 'in', 'amount', 'into', 'after', 'those', 'it', 'the', 'myself', 'out', 'at', 'something', 'nor', 'through', 'name', 'whereby', 'beforehand', 'are', 'nine', 'again', 'her', 'down', 'many', 'by', 'third', 'thence', 'becoming', 'a', 'found', 'five', 'however', 'nobody', 'such', 'three', 'latterly', 'keep', 'four', 'whereas', 're', 'have', 'since', 'someone', 'ourselves', 'inc', 'us', 'because', 'behind', 'wherever', 'for', 'below', 'am', 'somehow', 'with', 'each', 'former', 'six', 'still', 'couldnt', 'full', 'twenty', 'whither', 'otherwise', 'you', 'hereby', 'perhaps', 'hence', 'go', 'seem', 'sometime', 'onto', 'your', 'beside', 'me', 'then', 'had', 'their', 'against', 'anyone', 'and', 'himself', 'thru', 'hereafter', 'see', 'ever', 'give', 'was', 'yourselves', 'if', 'nevertheless', 'although', 'on', 'were', 'she', 'themselves', 'around', 'its', 'nowhere', 'part', 'as', 'throughout', 'get', 'towards', 'etc', 'fifty',  'empty', 'neither', 'most', 'via', 'wherein', 'thereupon', 'indeed', 'though', 'why', 'yours', 'moreover', 'becomes', 'un', 'whatever', 'whereupon', 'an', 'whence', 'everything', 'been', 'all', 'elsewhere', 'formerly', 'others', 'find', 'mostly', 'sincere', 'during', 'last', 'move', 'will',  'much', 'while', 'several', 'within', 'until', 'that', 'either', 'ie', 'latter', 'thin', 'may', 'together', 'fifteen', 'any', 'became', 'own', 'done', 'afterwards', 'also', 'he', 'besides', 'both', 'system', 'ten', 'toward', 'hers', 'not', 'two', 'hasnt', 'i', 'alone', 'has', 'only', 'upon', 'whose', 'thus', 'along', 'front', 'itself', 'even', 'is', 'should', 'now', 'among', 'first', 'herself', 'anywhere', 'made', 'next', 'herein', 'about', 'sometimes', 'whom', 'everywhere', 'least', 'fill', 'across', 'how','the']
context=[]
with open('train.json', 'r') as json_file:
	train = json.loads(json_file.read())
for i in range(len(train)):
    context.append(train[i][6])
    if(isTest == 'false'):
        looptimes = len(context) //size + 1


tfidf = TfidfVectorizer(lowercase=True,norm=None)#轉小寫，沒標準化，default是有標準化
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float64')
    return x 

X = tfidf_features(context, flag="train")
idf = tfidf.idf_

sentences_vec = []
for loop in range(looptimes):
    #sentences_token = np.load("sentences_token" + str(i) + ".npy", allow_pickle=True)
    #sentences_embedding = np.load("sentences_embedding" + str(i) + ".npy", allow_pickle=True)
    sentences_token = np.load(token_input_file + str(loop) + ".npy", allow_pickle=True)
    sentences_embedding = np.load(embedding_input_file +str(loop)+ ".npy", allow_pickle=True)

    for j in range(sentences_token.shape[0]):
        temp = np.zeros((768,))
        word_num = sentences_token[j].shape[0]
        for k in range(word_num):
            word = sentences_token[j][k]
            if(word in my_stop_words):
                continue
            idx = tfidf.vocabulary_.get(word)
            if(idx != None):
                temp += sentences_embedding[j][k] * idf[idx]
        if(word_num != 0):
            temp /= word_num
        sentences_vec.append(temp)
    
    del sentences_token
    del sentences_embedding

sentences_vec = np.array(sentences_vec)
print(sentences_vec.shape)
np.save(output_file, sentences_vec)