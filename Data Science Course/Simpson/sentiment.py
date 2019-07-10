import pandas as pd
import numpy as np

test_data_df = pd.read_csv('./X_test.csv')
test_data_class = pd.read_csv('./y_test.csv')
train_data_1 = pd.read_csv('./X_train.csv')
train_data_2 = pd.read_csv('./y_train.csv')
train_data_df = df = pd.merge(train_data_2, train_data_1, how="left", on="Unnamed: 0")

test_data_df = test_data_df.drop(['Unnamed: 0','id','spoken_words','speaking_line','raw_character_text'], axis=1)
train_data_df = train_data_df.drop(['Unnamed: 0','id','spoken_words','speaking_line','raw_character_text'], axis=1)

train_data_df.columns = ["character_id", "normalized_text"]
test_data_df.columns = ["normalized_text"]

#print(train_data_df.shape)
#print(test_data_df.shape)

import re, nltk
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):
	# remove non letters
	text = re.sub("[^a-zA-Z0-9]", " ", text)
	# tokenize
	tokens = nltk.word_tokenize(text)
	# stem
	stems = stem_tokens(tokens, stemmer)
	return stems

vectorizer = CountVectorizer(
	analyzer = 'word',
	tokenizer = tokenize,
	lowercase = True,
	stop_words = 'english',
	max_features = 100)


'''Generating significant features set'''
corpus_data_features = vectorizerfit_transform(train_data_df.normalized_text.tolist())
corpus_data_features_nd = corpus_data_features.toarray() # Convert feature sets to numpy array(numpy.ndarray)

print("="*50)
print(type(corpus_data_features_nd))
#print(len(corpus_data_features_nd))
print(corpus_data_features_nd[0])
print(corpus_data_features_nd[59665])


vocab = vectorizer.get_feature_names() # Get each features 
print(vocab)

''' ==========================================
 Print out each features and its count
dist = np.sum(corpus_data_features_nd, axis=0)
for tag, count in zip(vocab, dist):
	print(count, tag)
    ========================================== '''

# Split datasets to train and test sets


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

log_model = LogisticRegression()
#log_model.set_params(multi_class='multinomial', solver='newton-cg')
log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.character_id)
test_pred = log_model.predict(corpus_data_features_nd[:len(test_data_df)])
#corpus_data_features_nd[0:len(train_data_df)]

import random
spl = random.sample(range(len(test_pred)), 15)

for normalized_text, character_id in zip(test_data_df.normalized_text[spl], test_pred[spl]):
	print(character_id, normalized_text)

#test_data_df = map(lambda x: int(x), test_data_df)
#test_pred = map(lambda x: int(x), test_pred)
print(classification_report(test_data_class, test_pred))
