from ucimlrepo import fetch_ucirepo 

import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
import joblib

from data_preprocessing import DataPreprocessing

if __name__ == "__main__":

  # fetch dataset 
  bike_sharing = fetch_ucirepo(id=275) 

  # data (as pandas dataframes) 
  X = bike_sharing.data.features 
  y = bike_sharing.data.targets 

  # convert y to shape of (samples, )
  y = np.array(object=y)
  y = y.ravel()

  # Step 1: Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

  # Step 2: Define base_model
  # Step 3: Data preprocessing
  # Step 4: Create polynomial features
  # Step 5: Normalisation
  # Step 6: Use Recursive Feature Elimination + Linear Regression

  base_model = LinearRegression()

  pipe = Pipeline([('dropColumns', DataPreprocessing()),
                  ('poly', PolynomialFeatures(degree=3)), 
                  ('normalisation', StandardScaler(), ),
                  ('rfe', RFE(estimator=base_model, n_features_to_select=30))])
  
  pipe.fit(X_train, y_train)

  y_pred = pipe.predict(X_train)
  rmse = root_mean_squared_error(y_train, y_pred)
  print(f"RMSE of training set: {rmse}")

  y_pred = pipe.predict(X_test)
  rmse = root_mean_squared_error(y_test, y_pred)
  print(f"RMSE of testing set: {rmse}")

  joblib.dump(pipe, "model.pkl")

