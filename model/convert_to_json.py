import glob
import json
import os
import re
import nltk
import random
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')


def read_file(file):
    with open(file, 'r') as file:
        return file.read()


def write_json(file, data):
    with open(file, 'w') as fout:
        json.dump(data, fout, indent=4)


def preprocess_data(post):
    stemmer = WordNetLemmatizer()
    content = post

    # Remove all the special characters
    content = re.sub(r'\W', ' ', content)

    # Remove all single characters
    content = re.sub(r'\s+[a-zA-Z]\s+', ' ', content)

    # Remove single characters from the start
    content = re.sub(r'\^[a-zA-Z]\s+', ' ', content)

    # Substituting multiple spaces with single space
    content = re.sub(r'\s+', ' ', content, flags=re.I)

    # Removing prefixed 'b'
    content = re.sub(r'^b\s+', '', content)

    # Converting to Lowercase
    content = content.lower()

    # Lemmatization (cat's > cat)
    content = content.split()

    content = [stemmer.lemmatize(word) for word in content]
    content = ' '.join(content)

    # Remove Unicode characters
    content = content.encode("ascii", "ignore")
    return content.decode()


def predict_post(post):
    articles = ['business', 'entertainment', 'politics', 'sport', 'tech']
    data = preprocess_data(post)

    vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))
    with open('text_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    data = vectorizer.transform([data]).toarray()

    res = model.predict(data)

    return {articles[res[0]]}


def convert():
    # Get all directories in bbc folder
    list_of_categories = [f for f in os.scandir('../bbc') if f.is_dir()]
    X = []
    Y = []
    # Read files in folder preprocess data and append arrays with data and label
    for c_dir in list_of_categories:
        for file in glob.glob(str(c_dir.path) + "\\*.txt"):
            content = read_file(file)

            X.append(preprocess_data(content))
            Y.append(list_of_categories.index(c_dir))

    # Shuffle data
    Z = list(zip(X, Y))

    random.shuffle(Z)

    X, Y = zip(*Z)

    # min_df - cut off words that have a document frequency strictly higher than 5
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(X).toarray()

    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(classifier, picklefile)


if __name__ == '__main__':
    convert()
