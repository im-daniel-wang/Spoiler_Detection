import numpy as np
import pandas as pd
import pickle as pkl
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import argparse
import scipy as sp


MODEL = "svm.sav"
VECTORIZER = "vectorizer.pkl"

with open(VECTORIZER, 'rb') as f:
	TRANSFORMER = pkl.load(f)

SVM_MODEL = pkl.load(open(MODEL, 'rb'))

def process_sentence(text):
	"""
	process the input sentence word by word, removing stop words
	"""
	stop_words = set(stopwords.words('english')) 
	clean_text = [w.lower() for w in word_tokenize(text) if w not in stop_words and w.isalpha()]
	return ' '.join(clean_text)


def parse_review(text):
	num_sentences = len(text['review_sentences'])
	reviews = []
	perc_rank = []
	for i in range(len(text['review_sentences'])):
		reviews.append(process_sentence(text['review_sentences'][i]))
		perc_rank.append(i/num_sentences)
	return reviews, perc_rank


def predict(text):
	"""
	predict the class as spoiler or non-spoiler
	"""
	raw_reviews = text['review_sentences']
	reviews, perc_rank = parse_review(text)

	predicted_class = []

	if len(reviews):
		for i in range(len(reviews)):
			# tfidf transformation
			cleaned_text = TRANSFORMER.transform([reviews[i]])
			# append proportion scores
			cleaned_text = sp.sparse.hstack((cleaned_text, np.reshape([perc_rank[i]], (-1, 1))))

			pred = SVM_MODEL.predict(cleaned_text)
			
			if pred[0]==0:
				predicted_class.append('Non-Spoiler')
			else:
				predicted_class.append('Spoiler')
		res = {"reviews":raw_reviews, "Class":predicted_class}
		# res = {raw_reviews[i]:predicted_class[i] for i in range(len(predicted_class))}

		return json.dumps(res)


if __name__ == '__main__':

	text = {'review_sentences':['He died in the last episode.', 'I really enjoy this show.']}
	print(predict(text))


