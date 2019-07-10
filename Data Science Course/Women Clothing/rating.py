import pandas as pd
import numpy as np

print("Loading Data...\n")

df_train = pd.read_csv("./train.csv")
df_val = pd.read_csv("./val.csv")

# Keep "Review Text" and "Rating" columns in df
df_train = df_train[['Review Text', 'Rating']]
df_val = df_val[['Review Text', 'Rating']]

# Rename columns for future use
df_train.columns = ["review_text", "rating"]
df_val.columns = ["review_text", "rating"]

rate = [1, 2, 3, 4, 5]
index_to_drop = []

print("\nLoad Completed!\n")
print("\nCleaning Data...\n")

# Drop data in train.csv if entry in review_text and rating column is not valid
for i in range(len(df_train)):
	rates = df_train.iloc[i].loc['rating']
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
	rates = df_val.iloc[i].loc['rating']
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
y_train = df_train.rating
X_train1 = df_train.review_text
y_train1 = df_train.rating

X_val = df_val.review_text
y_val = df_val.rating

print("\nDrawing distribution histogram...\n")

# Draw distribution histogram
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(15,6)
pd.value_counts(y_train).plot.bar(width=1,align='center', alpha=1)
plt.title('Count histogram')
plt.xlabel('Rating')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
print(y_train.value_counts())
#plt.show()


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

count_vect = CountVectorizer(tokenizer=tokenize, 
							min_df=5, 
							encoding='utf-8', 
							ngram_range=(1, 2), 
							stop_words='english', 
							max_features=500)

tfidf = TfidfVectorizer(tokenizer=tokenize, 
						sublinear_tf=True,
						min_df=5, 
						norm='l2', 
						encoding='utf-8', 
						ngram_range=(1, 2), 
						stop_words='english', 
						max_features=1500)

#xtrain = tfidf.fit_transform(X_train).toarray()
#ytrain = tfidf.fit_transform(y_train).toarray()
#print(xtrain.shape)
#print(ytrain.shape)

#features = tfidf.fit_transform(X_train).toarray()
#print(features.shape)


print("\nLoading Test Data\n")

df_test = pd.read_csv("./test.csv")
df_test = df_test[["Review Text", "Rating"]]
df_test.columns = ["review_text", "rating"]
index_to_drop = []
for i in range(len(df_test)):
	rates = df_test.iloc[i].loc['rating']
	text = df_test.iloc[i].loc['review_text']
	try:
		if int(rates) not in rate or type(text) != str:
			index_to_drop.append(i)
	except ValueError:
		index_to_drop.append(i)
df_test.drop(index_to_drop, inplace=True)

X_test = df_test.review_text
y_test = df_test.rating

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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


##################################### Linear Regression ##############################################

lir = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
	('classification', LinearRegression())
	])

lir.fit(X_train, y_train)
test_pred = lir.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Linear Regression")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Logistic Regression regular ###############################

lor_reg = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('classification', LogisticRegression())
	])

lor_reg.fit(X_train, y_train)
test_pred = lor_reg.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression regular")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Logistic Regression with VarianceThreshold feature selection ###############################

lor_var = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', VarianceThreshold(threshold=.001)),
	('classification', LogisticRegression())
	])

lor_var.fit(X_train, y_train)
test_pred = lor_var.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with VarianceThreshold feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

####################### Logistic Regression with SelectKBest feature selection ########################################

lor_kbest = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectKBest(chi2, k=5)),
	('classification', LogisticRegression())
	])

lor_kbest.fit(X_train, y_train)
test_pred = lor_kbest.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with SelectKBest feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with LinearSVC feature selection################################

lor_li = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
	('classification', LogisticRegression())
	])

lor_li.fit(X_train, y_train)
test_pred = lor_li.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with LinearSVC feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with L1-based feature selection################################

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('predict', LassoCV(alphas=[1, 0.1, 0.001, 0.0005, 0.000001, 0.0000001, 0.000000001], max_iter=1e5)),
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with L1-based feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with L1-based feature selection (1) ################################

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('predict', Lasso(alpha=1, max_iter=15)),
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with L1-based feature selection(1)")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with L1-based feature selection (0.001) ################################

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('predict', Lasso(alpha=0.001, max_iter=15)),
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with L1-based feature selection(0.001)")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with L1-based feature selection (CV) ################################

features = tfidf.fit_transform(X_train).toarray()

lassocv = LassoCV()
lassocv.fit(features, y_train)
alpha = lassocv.alpha_

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('predict', Lasso(alpha=alpha, max_iter=15)),
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with L1-based feature selection(CV)")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################## Logistic Regression with L1-based feature selection (0.00001) ################################

lor_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('predict', Lasso(alpha=0.00001, max_iter=15)),
	])

lor_l1.fit(X_train, y_train)
test_pred = lor_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with L1-based feature selection(0.00001)")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################### Logistic Regression with Tree-based feature selection #####################################

lor_tree = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=30))),
	('classification', LogisticRegression())
	])

lor_tree.fit(X_train, y_train)
test_pred = lor_tree.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with Tree-based feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################### Logistic Regression with PCA feature selection #####################################

lor_pca = Pipeline([
	#('vect', tfidf),
	#('tfidf', TfidfTransformer()),
	('scaling', StandardScaler()),
	('pca', PCA(n_components=2)),
	('regr', LogisticRegression())
	])
xtrain = tfidf.fit_transform(X_train).toarray()
lor_pca.fit(xtrain, y_train)
xtest = tfidf.fit_transform(X_test).toarray()
test_pred = lor_pca.predict(xtest)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Logistic Regression with PCA feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Naive Bayes regular ###############################

nb_reg = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('classification', MultinomialNB())
	])

nb_reg.fit(X_train, y_train)
test_pred = nb_reg.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes regular")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

############################ Naive Bayes with VarianceThreshold feature selection ###############################

nb_var = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', VarianceThreshold(threshold=.001)),
	('classification', MultinomialNB())
	])

nb_var.fit(X_train, y_train)
test_pred = nb_var.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes with VarianceThreshold feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

####################### Naive Bayes with SelectKBest feature selection ########################################

nb_kbest = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectKBest(chi2, k=5)),
	('classification', MultinomialNB())
	])

nb_kbest.fit(X_train, y_train)
test_pred = nb_kbest.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes with SelectKBest feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

########################### Naive Bayes with L1-based feature selection ####################################

nb_l1 = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False))),
	('classification', MultinomialNB())
	])

nb_l1.fit(X_train, y_train)
test_pred = nb_l1.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes with L1-based feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################### Naive Bayes with Tree-based feature selection #####################################

nb_tree = Pipeline([
	('vect', tfidf),
	('tfidf', TfidfTransformer()),
	('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=30))),
	('classification', MultinomialNB())
	])

nb_tree.fit(X_train, y_train)
test_pred = nb_tree.predict(X_test)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes with Tree-based feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))

######################### Naive Bayes with PCA feature selection #####################################

nb_pca = Pipeline([
	#('vect', tfidf),
	#('tfidf', TfidfTransformer()),
	('scaling', StandardScaler()),
	('pca', PCA(n_components=2)),
	('regr', LogisticRegression())
	])
xtrain = tfidf.fit_transform(X_train).toarray()
nb_pca.fit(xtrain, y_train)
xtest = tfidf.fit_transform(X_test).toarray()
test_pred = nb_pca.predict(xtest)

y_val = map(lambda x: int(x), y_test)
test_pred = map(lambda x: int(x), test_pred)

print("="*60)
print("Naive Bayes with PCA feature selection")
print("="*60)

print(classification_report(y_test, test_pred))
print('Accuracy: %s\n' % accuracy_score(test_pred, y_test))