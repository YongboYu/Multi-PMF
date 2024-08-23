import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from darts import TimeSeries
from darts.metrics import mae as darts_mae, rmse as darts_rmse
from darts.dataprocessing.transformers import Scaler
from typing import List, Tuple

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

import os
import joblib

dataset = 'BPI2019_1'
# dataset = 'Hospital_Billing'
# dataset = 'RTFMP'

# %% Load the model
# Load the model from the file
# model_dir = f'saved_models/{dataset}/'
model_dir = f'trained_Darts_models/{dataset}/'

# List all files in the model directory
# model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

# Load each model and store them in a dictionary
trained_models = {}
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = os.path.splitext(model_file)[0]

    if 'RNN' in model_name:
        trained_models[model_name] = RNNModel.load(model_path)
    elif 'LSTM' in model_name:
        trained_models[model_name] = RNNModel.load(model_path)
    elif 'GRU' in model_name:
        trained_models[model_name] = RNNModel.load(model_path)
    elif 'N-BEATS' in model_name:
        trained_models[model_name] = NBEATSModel.load(model_path)
    elif 'N-HiTS' in model_name:
        trained_models[model_name] = NHiTSModel.load(model_path)
    elif 'TCN' in model_name:
        trained_models[model_name] = TCNModel.load(model_path)
    elif 'Transformer' in model_name:
        trained_models[model_name] = TransformerModel.load(model_path)

# Print loaded models
print(f"Loaded models: {list(trained_models.keys())}")

# %% extract the multivariate time series models
sub_models = [
    # 'Persistence',
    # 'Mean',
    # 'NaiveSeasonal',
    # 'NaiveDrift',
    # 'NaiveMovingAverage',
    'LinearRegression',
    'LightGBM',
    'XGBoost',
    # 'RNN',
    # 'LSTM',
    # 'GRU',
    # 'N-BEATS',
    # 'N-HiTS',
    # 'TCN',
    # 'Transformer',
    # 'TFT',
    # 'DLinear',
    # 'NLinear'
]

sub_trained_models = {key: trained_models[key] for key in sub_models if key in trained_models}

print(sub_trained_models)

# %% Load the test data
data = pd.read_hdf(f'./dataset/{dataset}.h5')
series = TimeSeries.from_dataframe(data)

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

# %% predict - DL
# Initialize a dictionary to store predictions for each model
all_model_predictions = {}

# Iterate over each model in trained_models
for model_name, model in trained_models.items():
    print(f'Predicting with {model_name} model...')

    # Initialize a list to store predictions for the current model
    model_predictions = []

    # model_predictions = model.predict(n=output_length, series=test_scaled_input_seq)

    # Loop through each TimeSeries object in test_scaled_input_seq
    for input_seq_ts in test_scaled_input_seq:
        # Use the model's predict function to predict the output for each input sequence
        pred_series = model.predict(n=output_length, series=input_seq_ts)
        pred_series_rescaled = scaler.inverse_transform(pred_series)

        # Append the predictions to the list
        model_predictions.append(pred_series_rescaled.values())

    # Store the predictions for the current model in the dictionary
    all_model_predictions[model_name] = np.array(model_predictions)

# Print the shape of predictions for each model
for model_name, predictions in all_model_predictions.items():
    print(f'{model_name} predictions shape: {predictions.shape}')

# save the predictions
np.savez_compressed(f'./result/{dataset}/new_mulit_predictions_rescaled.npz', **all_model_predictions)


# %% predict - Baseline (Persistence)
# Initialize the NaiveSeasonal model with K=1
model = NaiveSeasonal(K=1)

# Initialize a list to store predictions
persistence_predictions = []

# Iterate over each TimeSeries object in test_input_seq
for input_seq_ts in test_input_seq:
    # Fit the model to the input sequence
    model.fit(input_seq_ts)

    # Predict the output sequence length
    pred_series = model.predict(n=output_length)

    # Append the predictions to the list
    persistence_predictions.append(pred_series.values())

# Convert the list of predictions to a numpy array
persistence_predictions = np.array(persistence_predictions)

# Print the shape of the predictions
print(f'NaiveSeasonal predictions shape: {persistence_predictions.shape}')

# add the predictions to the dictionary
all_model_predictions['Persistence'] = persistence_predictions

# %% predict - Baseline (Mean)
# Initialize the NaiveSeasonal model with K=1
model = NaiveMean()

# Initialize a list to store predictions
mean_predictions = []

# Iterate over each TimeSeries object in test_input_seq
for input_seq_ts in test_input_seq:
    # Fit the model to the input sequence
    model.fit(input_seq_ts)

    # Predict the output sequence length
    pred_series = model.predict(n=output_length)

    # Append the predictions to the list
    mean_predictions.append(pred_series.values())

# Convert the list of predictions to a numpy array
mean_predictions = np.array(mean_predictions)

# Print the shape of the predictions
print(f'Mean predictions shape: {mean_predictions.shape}')

# add the predictions to the dictionary
all_model_predictions['Mean'] = mean_predictions

# %% try - NO
from darts import TimeSeries

# Initialize a dictionary to store predictions for each model
all_model_predictions = {}

# Ensure that test_scaled_input_seq is a list of TimeSeries objects
assert all(isinstance(seq, TimeSeries) for seq in test_scaled_input_seq), "All sequences must be TimeSeries objects"

# Iterate over each model in sub_trained_models
for model_name, model in sub_trained_models.items():
    print(f'Predicting with {model_name} model...')

    # Ensure the model is properly initialized
    if model.model is None:
        print(f"Model {model_name} is not properly initialized.")
        continue

    # Initialize a list to store predictions for the current model
    model_predictions = []

    # Loop through each TimeSeries object in test_scaled_input_seq
    for input_seq_ts in test_scaled_input_seq:
        # Use the model's predict function to predict the output for each input sequence
        pred_series = model.predict(n=output_length, series=input_seq_ts)

        # Append the predictions to the list
        model_predictions.append(pred_series.values())

    # Store the predictions for the current model in the dictionary
    all_model_predictions[model_name] = np.array(model_predictions)

# Print the shape of predictions for each model
for model_name, predictions in all_model_predictions.items():
    print(f'{model_name} predictions shape: {predictions.shape}')


# %% check models loading - NO
# Check if models are properly initialized
for model_name, model in sub_trained_models.items():
    if model.model is None:
        print(f"Model {model_name} is not properly initialized.")
    else:
        print(f"Model {model_name} is properly initialized.")


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
np.save(f'./result/{dataset}/new_mulit_pred_all.npy', all_model_metrics)


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
np.save(f'./result/{dataset}/new_last_day_pred_metrics.npy', last_day_model_metrics)


# %% plot the last day predictions

# Iterate over each model in all_model_predictions
for model_name, predictions in all_model_predictions.items():
    print(f'Plotting last-day predictions for {model_name} model...')

    # Initialize a figure for the current model
    plt.figure(figsize=(10, 6))

    # Loop through each prediction and corresponding ground truth
    for i in range(len(predictions)):
        # Extract the last-day prediction and ground truth
        pred_series = TimeSeries.from_values(predictions[i, -1, :])
        true_series = TimeSeries.from_values(transformed_output_seq_array[i, -1, :])

        # Plot the ground truth
        plt.plot(true_series.values(), label='Ground Truth', color='blue')

        # Plot the prediction
        plt.plot(pred_series.values(), label=f'Prediction {i+1}', linestyle='--')

    # Add title and labels
    plt.title(f'Last-day Predictions vs Ground Truth for {model_name} Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from darts import TimeSeries

# Create a PdfPages object to save all plots in a single PDF
pdf_pages = PdfPages(f'./result/{dataset}/last_day_predictions_plots.pdf')

# Iterate over each model in all_model_predictions
for model_name, predictions in all_model_predictions.items():
    print(f'Plotting last-day predictions for {model_name} model...')

    # Iterate over each feature (DF) in the predictions
    for feature_idx in range(predictions.shape[2]):
        # Create a new figure for the current feature
        plt.figure(figsize=(10, 6))

        # Loop through each day and plot the last-day predictions against the ground truth
        for day_idx in range(predictions.shape[0]):
            # Extract the last-day prediction and ground truth for the current feature
            pred_series_values = np.array([predictions[day_idx, -1, feature_idx]])
            true_series_values = np.array([transformed_output_seq_array[day_idx, -1, feature_idx]])

            # Ensure the values are numpy arrays
            pred_series = TimeSeries.from_values(pred_series_values)
            true_series = TimeSeries.from_values(true_series_values)

            # Plot the ground truth
            plt.plot(true_series.values(), label=f'Ground Truth Day {day_idx+1}', color='blue')

            # Plot the prediction
            plt.plot(pred_series.values(), label=f'Prediction Day {day_idx+1}', linestyle='--')

        # Add title and labels
        plt.title(f'Last-day Predictions vs Ground Truth for {model_name} Model - Feature {feature_idx+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # Save the figure to the PdfPages object
        pdf_pages.savefig()
        plt.close()

# Close the PdfPages object to finalize the PDF
pdf_pages.close()