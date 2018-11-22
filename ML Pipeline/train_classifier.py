import sys
import pandas as pd
import nltk
import re
import numpy as np
import pickle
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(database_filepath):
    '''

    :param database_filepath: the filepath to the database where the cleaned dataset was uploaded
    :return:
            X (df): our input message
            Y (df): the target categories
            category_names (list): a list of all the target category names
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categorized', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns.values.tolist()

    return X, Y, category_names

def tokenize(text):
    # We first normalize the test
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    # Then we remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # We instantiate our lemmatizer and apply it to all of our tokens
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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