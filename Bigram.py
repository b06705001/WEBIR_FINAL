import nltk
import math
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import random

def preprocessing(df):
	le = LabelEncoder()
	df[5] = le.fit_transform(df[5])

	review, label = df[6], df[5]

	pos_bigrams = []
	pos_unigrams = []
	neg_bigrams = []
	neg_unigrams = []

	for data, l in zip(review, label):
		data = data.lower().strip()
		for sent in sent_tokenize(data):
			token = ['<s>'] + word_tokenize(sent.strip(' ')) + ['</s>']
			bigram_list = list(nltk.bigrams(token))
			if l == 1:	
				pos_bigrams += list(bigram_list)
				pos_unigrams += list(token)
			else:
				neg_bigrams += list(bigram_list)
				neg_unigrams += list(token)

	return pos_bigrams, pos_unigrams, neg_bigrams, neg_unigrams

def saveModel(pos_bigram_freq, pos_unigram_freq, neg_bigram_freq, neg_unigram_freq):
	with open('pos_bigram_freq.pickle', 'wb') as f:
		pickle.dump(pos_bigram_freq, f)
	with open('pos_unigram_freq.pickle', 'wb') as f:
		pickle.dump(pos_unigram_freq, f)
	with open('neg_bigram_freq.pickle', 'wb') as f:
		pickle.dump(neg_bigram_freq, f)
	with open('neg_unigram_freq.pickle', 'wb') as f:
		pickle.dump(neg_unigram_freq, f)

def loadModel():
	with open('pos_bigram_freq.pickle', 'rb') as f:
		pos_bigram_freq = pickle.load(f)
	with open('pos_unigram_freq.pickle', 'rb') as f:
		pos_unigram_freq = pickle.load(f)
	with open('neg_bigram_freq.pickle', 'rb') as f:
		neg_bigram_freq = pickle.load(f)
	with open('neg_unigram_freq.pickle', 'rb') as f:
		neg_unigram_freq = pickle.load(f)
	return pos_bigram_freq, pos_unigram_freq, neg_bigram_freq, neg_unigram_freq

def getBigramFreq(unigrams, bigrams):
	unigram_freq = Counter(unigrams)
	stopword = list(map(lambda x: x[0], unigram_freq.most_common(10)))
	print(stopword)
	unk_unigram_freq = Counter()
	cnt = 0
	for word, time in unigram_freq.items():
		if unigram_freq[word] < 3:
			cnt += unigram_freq[word]
		else:
			unk_unigram_freq[word] = unigram_freq[word]
	unk_unigram_freq['<UNK>'] = cnt
	unk_bigrams = []
	for bigram in bigrams:
		if bigram[0] not in stopword or bigram[1] not in stopword:
			tmp = list(bigram)
			if bigram[0] not in unk_unigram_freq.keys():
				tmp[0] = '<UNK>'
			if bigram[1] not in unk_unigram_freq.keys():
				tmp[1] = '<UNK>'
			unk_bigrams.append(tuple(tmp))
	bigram_freq = Counter(unk_bigrams)
	return bigram_freq, unk_unigram_freq

def getBigramProb(bigram_freq, unigram_freq):
	word_size = len(unigram_freq)
	bigram_prob = {word: float(time+1)/(unigram_freq[word[0]]+word_size) for word, time in bigram_freq.items()}
	return bigram_prob

def getSentProb(bigram_list, bigram_prob, unigram_freq, stopword):
	prob = 0.0
	word_size = len(unigram_freq)
	for bigram in bigram_list:
		if bigram[0] not in stopword or bigram[1] not in stopword:
			tmp = list(bigram)	
			if bigram[0] not in unigram_freq.keys():
				tmp[0] = '<UNK>'
			if bigram[1] not in unigram_freq.keys():
				tmp[1] = '<UNK>'	
			tmp = tuple(tmp)
			if tmp in bigram_prob.keys():
				prob += math.log(bigram_prob[tmp], 2)
			else:
				prob += math.log(float(1)/(unigram_freq[tmp[0]]+word_size), 2)
	return prob

def predict(df, pos_bigram_prob, pos_unigram_freq, neg_bigram_prob, neg_unigram_freq):
	le = LabelEncoder()
	df[5] = le.fit_transform(df[5])
	test_data, y_true = df[6], df[5]
	y_pred = []
	pos_stopword = list(map(lambda x: x[0], pos_unigram_freq.most_common(20)))
	neg_stopword = list(map(lambda x: x[0], neg_unigram_freq.most_common(20)))
	for data, l in zip(test_data, y_true):
		data = data.lower().strip()
		pos_prob = 0.0
		neg_prob = 0.0
		for sent in sent_tokenize(data):
			sent = sent.lower()
			token = ['<s>'] + word_tokenize(sent) + ['</s>']
			bigram_list = list(nltk.bigrams(token))
			pos_prob += getSentProb(bigram_list, pos_bigram_prob, pos_unigram_freq, pos_stopword)
			neg_prob += getSentProb(bigram_list, neg_bigram_prob, neg_unigram_freq, neg_stopword)
		if pos_prob > neg_prob:
			y_pred.append(1)
		else:
			y_pred.append(0)
	return y_pred, y_true

def evaluate(y_pred, y_true):
	print(accuracy_score(y_true, y_pred))
	print(precision_score(y_true, y_pred, average='macro'))
	print(recall_score(y_true, y_pred, average='macro'))
	print(f1_score(y_true, y_pred, average='macro'))

'''
df_train = pd.read_json('train.json')
pos_bigrams, pos_unigrams, neg_bigrams, neg_unigrams = preprocessing(df_train)
pos_bigram_freq, pos_unigram_freq = getBigramFreq(pos_unigrams, pos_bigrams)
neg_bigram_freq, neg_unigram_freq = getBigramFreq(neg_unigrams, neg_bigrams)

saveModel(pos_bigram_freq, pos_unigram_freq, neg_bigram_freq, neg_unigram_freq)
'''
pos_bigram_freq, pos_unigram_freq, neg_bigram_freq, neg_unigram_freq = loadModel()

pos_bigram_prob = getBigramProb(pos_bigram_freq, pos_unigram_freq)
neg_bigram_prob = getBigramProb(neg_bigram_freq, neg_unigram_freq)

df_test = pd.read_json('test.json')
y_pred, y_true = predict(df_test, pos_bigram_prob, pos_unigram_freq, neg_bigram_prob, neg_unigram_freq)

evaluate(y_pred, y_true)


