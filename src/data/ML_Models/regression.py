from sklearn.linear_model import(
  LinearRegression,
  Ridge,
  Lasso,
  ElasticNet,
)

from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score,
)

from src.data.load_data import load_data

from src.data.preprocess import(
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

def create_model():
   models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
   }
   return models

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return{
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }