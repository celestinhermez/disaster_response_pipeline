from sklearn.base import BaseEstimator, TransformerMixin
from langdetect import detect, DetectorFactory
import pandas as pd

# We want to make the language detection deterministic
DetectorFactory.seed = 0

class MessageLanguage(BaseEstimator, TransformerMixin):

    def message_language(self, text):
        # We catch the exceptions where langdetect cannot detect a language and assume it is not english
        try:
            lang = detect(text)

            if lang == 'en':
                english = 1
            else:
                english = 0

        except:
            english = 0

        return english


    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_language = pd.Series(X).apply(self.message_language)
        return pd.DataFrame(X_language)