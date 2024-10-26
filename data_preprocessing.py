from sklearn.base import BaseEstimator, TransformerMixin

class DataPreprocessing(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.columns_to_drop = ['dteday', 'yr']
    
  def fit(self, X, y=None):
    return self  # No fitting necessary

  def transform(self, X):
    return X.drop(columns=self.columns_to_drop)