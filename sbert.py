import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SbertEmbedding(TransformerMixin, BaseEstimator):
    def __init__(self, model, batch_size=1, layer=-1):
        # From https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
        # For pickling reason you should not load models in __init__
        self.model = model
        self.layer = layer
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = SentenceTransformer(self.model)

        if isinstance(X, (pd.DataFrame, pd.Series)):
            nona = X.dropna()
            vectors_np = model.encode(nona.array).tolist()
            return pd.DataFrame(vectors_np, index=nona.index)
        else:
            return model.encode(X)

sbert = SbertEmbedding("all-MiniLM-L6-v2")
