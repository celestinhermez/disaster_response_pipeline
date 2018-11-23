from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import pandas as pd

class MessageLanguage(BaseEstimator, TransformerMixin):

    def message_language(self, text):
        blob = TextBlob(text)
        lang = blob.detect_language()

        if lang == 'en':
            english = 1
        else:
            english = 0

        return english


    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_language = pd.Series(X).apply(self.message_language)
        return pd.DataFrame(X_language)