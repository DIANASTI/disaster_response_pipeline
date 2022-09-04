# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import pickle
import re
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM data", engine)
    Y = df.values[:, df.shape[1]-36:df.shape[1]].astype('int')
    X = df.message.to_list()
    category_names = df.columns[4:].to_list()
    return X, Y, category_names


def tokenize(text):
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    best_params = {'clf__estimator__min_samples_split': 3, 'clf__estimator__n_estimators': 200, 'vect__ngram_range': (1, 2)}
    pipeline.set_params(**best_params)
    
    return pipeline
    
def display_results(cv, y_test, y_pred):
    labels=np.unique(y_pred)
    confusion_mat=confusion_matrix(y_test, y_pred, labels=labels)
    accuracy=(y_pred==y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
#    print("\nBest Parameters:", cv.best_params_)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(0, 36):
        print(category_names[i])
        #print(classification_report(y_test[:,i], y_pred[:,i]))
        display_results(model, Y_test[:,i], y_pred[:,i])


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()