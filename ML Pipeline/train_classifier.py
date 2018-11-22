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
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

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
    # We first build our pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # Then we iterate over different hyperparameters with GridSearch and return the GridSearch object
    parameters = {'clf__estimator': (RandomForestClassifier(), GradientBoostingClassifier(), SVC()),
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 0.75, 1.0)}
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # We obtain the predictions from our model
    y_pred = model.predict(X_test)

    # We create a loop to display the precision, recall, F1-score for each category as well as overall
    target_names = ['Negative', 'Positive']
    accuracy_list = []
    precision_list = []
    recall_list = []
    for i in range(y_pred.shape[1]):
        precision, recall, _, _ = precision_recall_fscore_support(Y_test.iloc[:, i], y_pred[:, i],
                                                                  average='weighted', target_names=target_names)
        accuracy = accuracy_score(Y_test.iloc[:, i], y_pred[:, i])
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)

        print('For the category: {}, \n'.format(category_names[i]))
        print('The accuracy is: {} \n'.format(accuracy))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))

    print('Overall accuracy is: {}'.format(np.array(accuracy).mean()))
    print('Overall precision is: {}'.format(np.array(precision).mean()))
    print('Overall recall is: {}'.format(np.array(recall).mean()))


def save_model(model, model_filepath):
    pickle.dump(model, model_filepath)


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