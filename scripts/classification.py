

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
import pickle

def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def preprocess_text(document):

    # remove numbers
    document = remove_numbers(document)

    # Remove punctuation
    document = document.translate(str.maketrans('', '', string.punctuation))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    document = document.strip()

    return document


test_proportion = 0.3
sampling_seed = 12
vectorizer_type= 'tfidf'
# vectorizer_type= 'count'

data0 = pd.read_csv(r"C:\SEIWork\python_projects\nlp\data\classification.csv")

data0['index'] = data0.index

data = data0.dropna(subset=['row_text'])
data = data[[True if len(i) > 0 else False for i in list(data['row_text'])]]

data['ptext'] = data['row_text'].apply(lambda x: preprocess_text(x))
data = data[[True if len(i) > 1 else False for i in list(data['ptext'])]]


corpus = list(data['ptext'])
y = data['row_label']

if vectorizer_type=='count':
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')
elif vectorizer_type=='tfidf':
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')

X_vector = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
dtm = X_vector.toarray()

X = pd.DataFrame(dtm, columns=feature_names, index=data.index)

print("data has total %s rows and %s columns" % (X.shape[0], X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=sampling_seed)

print("Train data has total %s rows and %s columns" % (X_train.shape[0], X_train.shape[1]))
print("Train data has total %s positive and %s negative" % (sum(y_train), len(y_train) - sum(y_train)))

print("Test data has total %s rows and %s columns" % (X_test.shape[0], X_test.shape[1]))
print("Test data has total %s positive and %s negative" % (sum(y_test), len(y_test) - sum(y_test)))


classifier = RandomForestClassifier(n_estimators=50, random_state=0, max_depth=20)
classifier.fit(X_train, y_train)

important_feaures = (list(
    sorted(
    zip(feature_names, list(map(lambda x: round(x, 4), classifier.feature_importances_))),
    key = lambda x: x[1],
    reverse=True)
))

y_pred = classifier.predict(X_test)

y_pred_coluumn_names = ['class_' + str(i) for i in classifier.classes_]

y_test_pred_prob = pd.DataFrame(classifier.predict_proba(X_test), columns=y_pred_coluumn_names, index = y_test.index)
y_test_pred_prob['index'] = y_test.index
y_test_pred_prob['t_flag'] = 'test'

y_train_pred_prob = pd.DataFrame(classifier.predict_proba(X_train), columns=y_pred_coluumn_names, index = y_train.index)
y_train_pred_prob['index'] = y_train.index
y_train_pred_prob['t_flag'] = 'train'

y_all = y_train_pred_prob.append(y_test_pred_prob)
y_all = y_all.sort_values(by=['index'], ascending=[True])

prob_th = 0.5
class_to_be_pred = 'class_1'

y_all['predicted_class'] = y_all[class_to_be_pred].apply(lambda x: 1 if x>prob_th else 0)

data_out = pd.merge(data0, y_all, how='left', on = 'index')

important_features = pd.DataFrame(important_feaures, columns=['feature', 'score'])

data_out.to_csv(r"C:\SEIWork\python_projects\nlp\data\data_out_30.csv", index=False)
important_features.to_csv(r"C:\SEIWork\python_projects\nlp\data\important_features.csv", index=False)

print("Test error rate %s \n" % round((1- accuracy_score(y_test, y_pred)),3))
print("Confusion matrix %s \n" % pd.DataFrame(confusion_matrix(y_test,y_pred), columns=y_test.unique(), index = y_test.unique()))
print(classification_report(y_test, y_pred))
