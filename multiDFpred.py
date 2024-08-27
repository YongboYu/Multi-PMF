import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from darts import TimeSeries
from darts.metrics import mae as darts_mae, rmse as darts_rmse
from darts.dataprocessing.transformers import Scaler
from typing import List, Tuple

import pickle
import os
import h5py

from darts.models import (
    NaiveSeasonal,
    NaiveMean,
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

# %% Load the dataset
import argparse

parser = argparse.ArgumentParser(description='Process some variables.')
parser.add_argument('--dataset', type=str, default='default_value', help='A variable to pass to the script')
args = parser.parse_args()

dataset = args.dataset

data = pd.read_hdf(f'./dataset/{dataset}.h5')

series = TimeSeries.from_dataframe(data)

# %% split the data
train, test = series.split_after(0.8)

scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# %% generate seq2seq data for test and test_scaled

# Define input and output sizes
input_length = 32
output_length = 16


def create_seq2seq_dataset(series: TimeSeries, input_length: int, output_length: int) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    input_series_list = []
    output_series_list = []

    for i in range(len(series) - input_length - output_length + 1):
        input_series = series[i:i + input_length]
        output_series = series[i + input_length:i + input_length + output_length]

        input_series_list.append(input_series)
        output_series_list.append(output_series)

    return input_series_list, output_series_list

test_input_seq, test_output_seq = create_seq2seq_dataset(test, input_length, output_length)
test_scaled_input_seq, test_scaled_output_seq = create_seq2seq_dataset(test_scaled, input_length, output_length)


# %% define models
models = [
    ('Persistence', NaiveSeasonal(K=1)),
    ('Mean', NaiveMean()),
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

# %% train and save the models
trained_models = {}
predictions = {}
metrics = []
models_directory = f'./trained__models/{dataset}/'

for name, model in models:
    print(f'Training {name} model...')
    model.fit(train_scaled)
    trained_models[name] = model
    # Save model
    filename = models_directory + name + '.pt'
    model.save(filename)
    print(f'Model {name} saved to {filename}')
    
    
# %% predict 
# Initialize a dictionary to store predictions for each model
all_model_predictions = {}

# Iterate over each model in trained_models
for name, model in trained_models.items():
    print(f'Predicting with {name} model...')

    # Initialize a list to store predictions for the current model
    model_predictions = []

    # Loop through each TimeSeries object in test_scaled_input_seq
    for input_seq_ts in test_scaled_input_seq:
        # Use the model's predict function to predict the output for each input sequence
        pred_series = model.predict(n=output_length, series=input_seq_ts)
        pred_series_rescaled = scaler.inverse_transform(pred_series)

        # Append the predictions to the list
        model_predictions.append(pred_series_rescaled.values())

    # Store the predictions for the current model in the dictionary
    all_model_predictions[name] = np.array(model_predictions)

# Print the shape of predictions for each model
for name, predictions in all_model_predictions.items():
    print(f'{name} predictions shape: {predictions.shape}')

# save the predictions
np.savez_compressed(f'./result/{dataset}/mulit_predictions.npz', **all_model_predictions)

# %% transform the test output ground truth

# Initialize a list to store the transformed sequences
transformed_output_seq = []

# Iterate over each TimeSeries object in test_output_seq
for output_seq_ts in test_output_seq:
    # Convert each TimeSeries object to a numpy array
    output_seq_array = output_seq_ts.values()
    # Append the numpy array to the list
    transformed_output_seq.append(output_seq_array)

# Convert the list of numpy arrays to a 3D numpy array
transformed_output_seq_array = np.array(transformed_output_seq)

# Print the shape to verify
print(transformed_output_seq_array.shape)

# %% evaluate the all predictions

# Initialize a dictionary to store evaluation metrics for each model
all_model_metrics = {}

# Iterate over each model in all_model_predictions
for model_name, predictions in all_model_predictions.items():
    print(f'Evaluating {model_name} model...')

    # Initialize lists to store metrics for the current model
    model_mae = []
    model_rmse = []

    # Loop through each prediction and corresponding ground truth
    for i in range(len(predictions)):
        pred_series = TimeSeries.from_values(predictions[i])
        true_series = TimeSeries.from_values(transformed_output_seq_array[i])

        # Compute MAE and RMSE
        mae = darts_mae(true_series, pred_series)
        rmse = darts_rmse(true_series, pred_series)

        # Append the metrics to the lists
        model_mae.append(mae)
        model_rmse.append(rmse)

    # Store the average metrics for the current model in the dictionary
    all_model_metrics[model_name] = {
        'MAE': np.mean(model_mae),
        'RMSE': np.mean(model_rmse)
    }

# Print the evaluation metrics for each model
for model_name, metrics in all_model_metrics.items():
    print(f'{model_name} model: MAE = {metrics["MAE"]:.2f}, RMSE = {metrics["RMSE"]:.2f}')

# save the metrics
np.save(f'./result/{dataset}/all_pred_metrics.npy', all_model_metrics)

# %% evaluate the last prediction

# Initialize a dictionary to store evaluation metrics for each model
last_day_model_metrics = {}

# Iterate over each model in all_model_predictions
for model_name, predictions in all_model_predictions.items():
    print(f'Evaluating last-day predictions for {model_name} model...')

    # Initialize lists to store metrics for the current model
    model_mae = []
    model_rmse = []

    # Loop through each prediction and corresponding ground truth
    for i in range(len(predictions)):
        # Extract the last-day prediction and ground truth
        pred_series = TimeSeries.from_values(predictions[i, -1, :])
        true_series = TimeSeries.from_values(transformed_output_seq_array[i, -1, :])

        # Compute MAE and RMSE
        mae = darts_mae(true_series, pred_series)
        rmse = darts_rmse(true_series, pred_series)

        # Append the metrics to the lists
        model_mae.append(mae)
        model_rmse.append(rmse)

    # Store the average metrics for the current model in the dictionary
    last_day_model_metrics[model_name] = {
        'MAE': np.mean(model_mae),
        'RMSE': np.mean(model_rmse)
    }

# Print the evaluation metrics for each model
for model_name, metrics in last_day_model_metrics.items():
    print(f'{model_name} model: Last-day MAE = {metrics["MAE"]:.2f}, Last-day RMSE = {metrics["RMSE"]:.2f}')

# Save the metrics
np.save(f'./result/{dataset}/last_pred_metrics.npy', last_day_model_metrics)

