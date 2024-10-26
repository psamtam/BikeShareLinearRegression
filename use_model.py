import joblib

from ucimlrepo import fetch_ucirepo 

from sklearn.metrics import root_mean_squared_error

if __name__ == "__main__":
  model = joblib.load("model.pkl")

  bike_sharing = fetch_ucirepo(id=275)

  X = bike_sharing.data.features 
  y = bike_sharing.data.targets 

  y_pred = model.predict(X)

  rmse = root_mean_squared_error(y, y_pred)
  print(f"RMSE is {rmse}")