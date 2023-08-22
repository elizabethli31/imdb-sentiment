import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer

test_csv = pd.read_csv('data/Test.csv')
train_csv = pd.read_csv('data/Train.csv')

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# lemmatize reviews
train_x_text = train_csv['text']
train_y = train_csv['label']
train_x = []

test_x_text = test_csv['text']
test_y = test_csv['label']
test_x = []

nltk.download('wordnet')
for i in range(0, len(train_x_text)):
    review = re.sub('[^a-zA-Z]', ' ', train_x_text[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_x.append(review)

for i in range(0, len(test_x_text)):
    review = re.sub('[^a-zA-Z]', ' ', test_x_text[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_x.append(review)

# tf-idf
tfidf = TfidfVectorizer()
x_train_tf = tfidf.fit_transform(train_x)
x_test_tf = tfidf.transform(test_x)
print("n_samples: %d, n_features: %d" % x_train_tf.shape)
print("n_samples: %d, n_features: %d" % x_test_tf.shape)


