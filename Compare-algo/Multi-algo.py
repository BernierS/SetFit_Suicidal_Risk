"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: https://github.com/BernierS/SetFit_Suicidal_Risk
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    This file is used to compare the accuracy of different machine learning algorithms on the dataset.
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer

# Load the dataset
df = pd.read_csv('Data/Suicide_Data_E16.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer()
# Fit the vectorizer on the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Train a Multinomial Naive Bayes classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

# Train a Random Forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_tfidf, y_train)

# Train a Logistic Regression classifier
lr_clf = LogisticRegression()
lr_clf.fit(X_train_tfidf, y_train)

# Train an SVM classifier
svm_clf = SVC()
svm_clf.fit(X_train_tfidf, y_train)

# Train an MLP classifier
mlp_clf = MLPClassifier()
mlp_clf.fit(X_train_tfidf, y_train)

# Train an Gradient classifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train_tfidf, y_train)

# Train an MLP classifier
pl_clf = Perceptron()
pl_clf.fit(X_train_tfidf, y_train)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert the text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to a fixed length
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Train an LSTM classifier
lstm_clf = Sequential()
lstm_clf.add(Embedding(len(tokenizer.word_index)+1, 100, input_length=max_len))
lstm_clf.add(LSTM(100))
lstm_clf.add(Dense(1, activation='sigmoid'))
lstm_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_clf.fit(X_train_pad, y_train, epochs=10, batch_size=32)

# Make predictions on the test data
nb_pred = nb_clf.predict(tfidf.transform(X_test))
rf_pred = rf_clf.predict(tfidf.transform(X_test))
lr_pred = lr_clf.predict(tfidf.transform(X_test))
svm_pred = svm_clf.predict(tfidf.transform(X_test))
mlp_pred = mlp_clf.predict(tfidf.transform(X_test))
gb_pred = gb_clf.predict(tfidf.transform(X_test))
pl_pred = pl_clf.predict(tfidf.transform(X_test))
lstm_pred = (lstm_clf.predict(X_test_pad) > 0.5).astype("int32")

# Calculate the accuracy of the models
nb_accuracy = accuracy_score(y_test, nb_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
pl_accuracy = accuracy_score(y_test, pl_pred)
lstm_accuracy = accuracy_score(y_test, lstm_pred)
print('Naive Bayes Accuracy:', nb_accuracy)
print('Random Forest Accuracy:', rf_accuracy)
print('Logistic Regression Accuracy:', lr_accuracy)
print('SVM Accuracy:', svm_accuracy)
print('MLP Accuracy:', mlp_accuracy)
print('GB Accuracy:', gb_accuracy)
print('PL Accuracy:', pl_accuracy)
print('LSTM Accuracy:', lstm_accuracy)