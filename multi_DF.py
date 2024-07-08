import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries

from darts.models import (
    NaiveSeasonal,
    NaiveMean,
    NaiveDrift,
    NaiveMovingAverage,
    KalmanForecaster,
    LinearRegressionModel,
    RandomForest,
    LightGBMModel,
    XGBModel,
    RNNModel,
    NBEATSModel,
    NHiTSModel,
    TCNModel,
    TransformerModel,
    TFTModel,
    DLinearModel,
    NLinearModel
)
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler


# %% import data

dataset = 'BPI2019_1'

# Load the dataset
data = pd.read_hdf(f'./dataset/{dataset}.h5')

series = TimeSeries.from_dataframe(data)

series.plot()
plt.legend()
plt.show()


# %% split and scale data
train, test = series.split_after(0.8)

scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# %% a series of multivariate time series models
# NaiveBaseline
# NaiveMean
# NaiveSeasonal
# NaiveDrift
# NaiveMovingAverage
# KalmanFilter
# LinearRegression
# RandomForest
# LightGBM
# XGBoost
# RNNModel: RNN, LSTM, GRU
# NBETS
# NHiTS
# TCN
# Transfomer
# TFT
# DLinear
# NLinear
models = [
    ('Persistence', NaiveSeasonal(K=1)),
    ('Mean', NaiveMean()),
    ('NaiveSeasonal', NaiveSeasonal(K=7)),
    ('NaiveDrift', NaiveDrift()),
    ('NaiveMovingAverage', NaiveMovingAverage(input_chunk_length=7)),
    ('KalmanFilter', KalmanForecaster(dim_x=149)),
    ('LinearRegression', LinearRegressionModel(
        lags=32,
        output_chunk_length=16
    )),
    ('RandomForest', RandomForest(
        lags=32,
        output_chunk_length=16,
        n_estimators=200,
        criterion="absolute_error"
    )),
    ('LightGBM', LightGBMModel(
        lags=32,
        output_chunk_length=16,
        verbose=1
    )),
    ('XGBoost', XGBModel(
        lags=32,
        output_chunk_length=16
    )),
    ('RNN', RNNModel(
        model="RNN",
        hidden_dim=64,
        n_rnn_layers=2,
        input_chunk_length=32,
        output_chunk_length=16,
        training_length=64,
        n_epochs=100,
    )),
    ('LSTM', RNNModel(
            model="LSTM",
            hidden_dim = 64,
            n_rnn_layers =2,
            input_chunk_length=32,
            output_chunk_length=16,
            training_length=64,
            n_epochs=100,
    )),
    ('GRU', RNNModel(
        model="GRU",
        hidden_dim = 64,
        n_rnn_layers =2,
        input_chunk_length=32,
        output_chunk_length=16,
        training_length=64,
        n_epochs=100,
    )),
    ('N-BEATS', NBEATSModel(
        input_chunk_length=32,
        output_chunk_length=16,
        generic_architecture=True,
        num_blocks=3,
        num_layers=4,
        layer_widths=256,
        n_epochs=100
    )),
    ('N-HiTS', NHiTSModel(
        input_chunk_length=32,
        output_chunk_length=16,
        num_blocks=3,
        num_layers=4,
        layer_widths=256,
        n_epochs=100
    )),
    ('TCN', TCNModel(
        input_chunk_length=32,
        output_chunk_length=16,
        n_epochs=100
    )),
    ('Transformer', TransformerModel(
        input_chunk_length=32,
        output_chunk_length=16,
        n_epochs=100
    )),
    ('TFT', TFTModel(
        input_chunk_length=32,
        output_chunk_length=16,
        n_epochs=100,
        add_relative_index =True
    )),
    ('DLinear', DLinearModel(
        input_chunk_length=32,
        output_chunk_length=16,
        n_epochs=100
    )),
    ('NLinear', NLinearModel(
        input_chunk_length=32,
        output_chunk_length=16,
        n_epochs=100
    ))
]



# %% train the models
trained_models = {}
for name, model in models:
    print(f'Training {name} model...')
    try:
        # Check if the model's fit method accepts a 'verbose' parameter
        # Attempt to fit with verbose=True
        model.fit(train_scaled, verbose=True)
    except TypeError:
        if 'verbose' in model.fit.__code__.co_varnames:
            # If TypeError is raised, fit without verbose
            model.fit(train_scaled, verbose=True)
            model.fit(train_scaled)
        else:
            model.fit(train_scaled)
    trained_models[name] = model

# %% predict and evaluate
predictions = []
metrics = []

for name, model in trained_models.items():
    print(f'Evaluating {name} model...')
    pred_series = model.predict(len(test_scaled), series=train_scaled)

    # Calculate metrics
    mae = mae(test_scaled, pred_series)
    rmse = rmse(test_scaled, pred_series)
    print(f'{name} model: MAE = {mae:.2f}, RMSE = {rmse:.2f}')

    # Store predictions
    pred_df = pred_series.pd_dataframe()
    pred_df.columns = [f"{col}_{name}" for col in pred_df.columns]
    predictions.append(pred_df)

    # Store metrics
    metrics.append((name, mae, rmse))

# Concatenate all predictions
all_predictions = pd.concat(predictions, axis=1)

# Create a metrics DataFrame
metrics_df = pd.DataFrame(metrics, columns=['Model', 'MAE', 'RMSE'])

# Print the metrics
print(metrics_df)


# %% save the predictions and metrics
# Save predictions to CSV
all_predictions.to_csv(f'result/{dataset}_predictions.csv')

# Save metrics to CSV
metrics_df.to_csv(f'result/{dataset}_metrics.csv')

# Print file paths
print(f"Predictions of {dataset} saved to predictions.csv")
print(f"Metrics of {dataset} saved to metrics.csv")







#
# # %% RNN model
# LSTM_model = RNNModel(
#     model="LSTM",
#     hidden_dim = 64,
#     n_rnn_layers =2,
#     input_chunk_length=32,
#     output_chunk_length=16,
#     training_length=64,
#     n_epochs=100,
# )
# LSTM_model.fit(train_scaled, verbose=True)
#
# # %% Predict
# # Make predictions
# pred_series = LSTM_model.predict(len(test_scaled), series=train_scaled)
#
# # Evaluate the model
# error = mae(test_scaled, pred_series)
# print(f'MAE: {error:.2f}%')
#
# # %% Plot
# # Plot the results
# plt.figure(figsize=(12, 8))
# series.plot(label='Actual')
# pred_series.plot(label='Forecast')
# plt.legend()
# plt.show()
#
#
# # %% lightGBM
# from darts.models import LightGBMModel
#
# model = LightGBMModel(
#     n_estimators=100,
#     num_leaves=10,
#     model_name="LightGBM",
# )
# model.fit(train_scaled, verbose=True)
#
# # %% VARIMA --> failed: Matrix is not positive definite
# model = VARIMA(p=3, d=0, q = 2)
# model.fit(train_scaled)
#
# # %% KalmanFilter
# from darts.models import KalmanForecaster
#
# model = KalmanForecaster(
#     dim_x=149
# )
# model.fit(train_scaled)