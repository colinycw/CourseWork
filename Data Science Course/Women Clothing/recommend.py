import pandas as pd
import numpy as np

print("Loading Data...\n")

df_train = pd.read_csv("./train.csv")
df_val = pd.read_csv("./val.csv")


# Keep "Review Text" and "Rating" columns in df
df_train = df_train[['Review Text', 'Recommended IND']]
df_val = df_val[['Review Text', 'Recommended IND']]

# Rename columns for future use
df_train.columns = ["review_text", "recommended"]
df_val.columns = ["review_text", "recommended"]

rate = [0, 1]
index_to_drop = []

print("\nLoad Completed!\n")
print("\nCleaning Data...\n")

# Drop data in train.csv if entry in review_text and rating column is not valid
for i in range(len(df_train)):
	rates = df_train.iloc[i].loc['recommended']
	text = df_train.iloc[i].loc['review_text']
	try:
		if int(rates) not in rate or type(text) != str:
			index_to_drop.append(i)
	except ValueError:
		index_to_drop.append(i)
'''index_to_drop1 = df_train[df_train['rating'] == 'NaN'].index
index_to_drop2 = df_train[df_train['review_text'] == 'NaN'].index
df_train.drop(index_to_drop1, inplace=True)
df_train.drop(index_to_drop2, inplace=True)'''
df_train.drop(index_to_drop, inplace=True)
df_train = df_train[pd.notnull(df_train['review_text'])]

index_to_drop = []

# Drop data in val.csv if entry in review_text and rating column is not valid
for i in range(len(df_val)):
	rates = df_val.iloc[i].loc['recommended']
	text = df_val.iloc[i].loc['review_text']
	try:
		if int(rates) not in rate or type(text) != str:
			index_to_drop.append(i)
	except ValueError:
		index_to_drop.append(i)
df_val.drop(index_to_drop, inplace=True)

print("\nClean Completed!\n")

# Convert original text into normalized text
# remove numbers or non-characters 
import re
from pattern.en import lemma
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

X_train = df_train.review_text
y_train = df_train.recommended

X_val = df_val.review_text
y_val = df_val.recommended

# print("\nDrawing distribution histogram...\n")

# # Draw distribution histogram
# import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize']=(15,6)
# pd.value_counts(y_train).plot.bar(width=1, align='center', alpha=1)
# plt.title('Count histogram')
# plt.xlabel('Recommeded IND')
# plt.xticks(rotation=0)
# plt.ylabel('Frequency')
# print(y_train.value_counts())
# plt.show()


def tokenize(text):
	lemaed = []
	# remove non letters
	text = re.sub("[^a-zA-Z]", " ", text)
	# tokenize
	tokens = TextBlob(text)
	# lemmatize
	for word in tokens.words:
		lemaed.append(word.lemmatize())

	return lemaed

# count_vect = CountVectorizer(tokenizer=tokenize,
# 							min_df=5,
# 							encoding='utf-8',
# 							ngram_range=(1, 2),
# 							stop_words='english',
# 							max_features=1500)

tfidf = TfidfVectorizer(tokenizer=tokenize, 
						sublinear_tf=True, 
						min_df=5, 
						norm='l2', 
						encoding='utf-8', 
						ngram_range=(1,4), 
						stop_words='english', 
						max_features=1000)

#xtrain = tfidf.fit_transform(X_train).toarray()
#ytrain = tfidf.fit_transform(y_train).toarray()
#print(xtrain.shape)
#print(ytrain.shape)

#features = tfidf.fit_transform(X_train).toarray()
#print(features.shape)


print("\nLoading Test Data\n")

df_test = pd.read_csv("./test.csv")
df_test = df_test[["Review Text", "Recommended IND"]]
df_test.columns = ["review_text", "recommended"]
index_to_drop = []
for i in range(len(df_test)):
	rates = df_test.iloc[i].loc['recommended']
	text = df_test.iloc[i].loc['review_text']
	try:
		if int(rates) not in rate or type(text) != str:
			index_to_drop.append(i)
	except ValueError:
		index_to_drop.append(i)
df_test.drop(index_to_drop, inplace=True)
df_test = df_test[pd.notnull(df_test['review_text'])]

X_test = df_test.review_text
y_test = df_test.recommended

print("\nRunning Model...\n")

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

########################## Logistic Regression regular #############################

print("="*60)
print("Logistic Regression regular")
print("="*60)

lor_reg = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('classification', LogisticRegression())
	])

lor_reg.fit(X_train, y_train)
test_pred = lor_reg.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############## Logistic Regression with VarianceThreshold feature selection ############

print("="*60)
print("Logistic Regression with VarianceThreshold feature selection")
print("="*60)

lor_var = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', VarianceThreshold(threshold=.0008)),
	('classification', LogisticRegression())
	])

lor_var.fit(X_train, y_train)
test_pred = lor_var.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############## Logistic Regression with SelectKBest feature selection ############

print("="*60)
print("Logistic Regression with SelectKBest feature selection")
print("="*60)

lor_kbest = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectKBest(chi2, k=5)),
	('classification', LogisticRegression())
	])

lor_kbest.fit(X_train, y_train)
test_pred = lor_kbest.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############## Logistic Regression with L1-based feature selection ############

print("="*60)
print("Logistic Regression with L1-based feature selection")
print("="*60)

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False))),
	('classification', LogisticRegression())
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############## Logistic Regression with Tree-based feature selection ############

print("="*60)
print("Logistic Regression with Tree-based feature selection")
print("="*60)

lor_tree = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=30))),
	('classification', LogisticRegression())
	])

lor_tree.fit(X_train, y_train)
test_pred = lor_tree.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Linear Support Vector Machine ###############################

print("="*60)
print("Linear Support Vector Machine")
print("="*60)

# Linear Support Vector Machine
sgd = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
	('classification', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=0, max_iter=5, tol=None))
	])
sgd.fit(X_train, y_train)
test_pred = sgd.predict(X_test)

# y_val = map(lambda x: int(x), y_val)
# test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Naive Bayes regular ###############################

print("="*60)
print("Naive Bayes regular")
print("="*60)

# Naive Bayes 
nb_reg = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('classification', MultinomialNB())
	])

nb_reg.fit(X_train, y_train)
test_pred = nb_reg.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############ Naive Bayes with VarianceThreshold feature selection ############

print("="*60)
print("Naive Bayes with VarianceThreshold feature selection")
print("="*60)

# Naive Bayes 
nb_var = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', VarianceThreshold(threshold=.0005)),	
	('classification', MultinomialNB())
	])

nb_var.fit(X_train, y_train)
test_pred = nb_var.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############ Naive Bayes with SelectKBest feature selection ############

print("="*60)
print("Naive Bayes with SelectKBest feature selection")
print("="*60)

# Naive Bayes 
nb_kbest = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectKBest(chi2, k=5)),
	('classification', MultinomialNB())
	])

nb_kbest.fit(X_train, y_train)
test_pred = nb_kbest.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############ Naive Bayes with L1-based feature selection ############

print("="*60)
print("Naive Bayes with L1-based feature selection")
print("="*60)

# Naive Bayes 
nb_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty ="l1", dual=False))),
	('classification', MultinomialNB())
	])

nb_l1.fit(X_train, y_train)
test_pred = nb_l1.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############ Naive Bayes with Tree-based feature selection ############

print("="*60)
print("Naive Bayes with Tree-based feature selection")
print("="*60)

# Naive Bayes 
nb_tree = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=30))),
	('classification', MultinomialNB())
	])

nb_tree.fit(X_train, y_train)
test_pred = nb_tree.predict(X_test)

#y_val = map(lambda x: int(x), y_test)
#test_pred = map(lambda x: int(x), test_pred)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))




