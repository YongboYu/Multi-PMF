# Multi-PMF: Multivariate Approaches for Process Model Forecasting

A framework for time series forecasting of business process models. This project combines process model forecasting techniques with advanced machine learning and deep learning models to predict process behavior.

## Overview

This project provides tools for:
- Preprocessing event logs
- Generating daily process model (Directly-Follows Graph, DFG)
- Converting process model to time series data
- Training and evaluating multiple forecasting models

## Project Structure

```
.
├── event_logs/           # Raw and processed event logs
├── data/                 # Daily DFGs and activity mapping
├── dataset/              # Processed time series datasets
├── lib/                  # Library files
├── preprocess.py         # Event log preprocessing
├── gen_daily_dfg.py      # Generates daily DFG matrics
├── dfg_to_time_series.py # Converts DFGs to DF time series
└── multiDFpred.py        # Main prediction script
```

## Features

- **Event Log Preprocessing**
  - Filters infrequent process variants
  - Trims event logs to specified time ranges
  - Adds artificial start/end activities

- **DF Time Series Generation**
  - Extracts DFGs with activity mapping
  - Time series conversion

- **Multiple Forecasting Models**
  - Baseline models:
    - Persistence
    - Mean
  - Advanced models:
    - Linear Regression
    - XGBoost
    - LightGBM
    - RNN/LSTM/GRU
    - N-BEATS
    - N-HiTS
    - TCN
    - Transformer
    - TFT
    - DLinear
    - NLinear

## Requirements

- Python 3.9
- PyTorch
- Darts
- PM4Py
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- h5py

## Usage

1. **Preprocess Event Logs**
```bash
python preprocess.py
```

2. **Generate Daily DFGs**
```bash
python gen_daily_dfg.py
```

3. **Convert to DF Time Series**
```bash
python dfg_to_time_series.py
```

4. **Run Predictions**
```bash
python multiDFpred.py --dataset <dataset_name>
```

## Dataset Setup
1. Download the event logs from the following sources:
   - [BPI2019_1](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853) - BPI Challenge 2019
   - [Hospital_Billing](https://data.4tu.nl/articles/dataset/Hospital_Billing_-_Event_Log/12705113) - Hospital Billing Event Log
   - [RTFMP](https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249) - Road Traffic Fine Management Process

2. Create the required directory structure:
   ```bash
   mkdir -p event_logs/original
   ```

3. Place the downloaded `.xes` files in the `event_logs/original` directory

4. After running `python preprocess.py`, the processed event logs will be saved in the `./event_logs/processed` directory.

## Model Training

The framework select the following parameters:
- Input sequence length: 32 (days)
- Output sequence length: 16 (days)
- Training/Test split: 80/20

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

## Repository

For the complete repository for PMF benchmark including additional resources, documentation, and examples, please visit:
[PMF-Benchmark](https://github.com/YongboYu/PMF-Benchmark)

## Reference

De Smedt, J., Yeshchenko, A., Polyvyanyy, A., De Weerdt, J., & Mendling, J. (2023). 
Process model forecasting and change exploration using time series analysis of event sequence data. 
Data & Knowledge Engineering, 145, 102145.

## Citation

If you find this repository helpful for your work, please consider citing our paper:

```bibtex
@inproceedings{yu2024multivariate,
  title={Multivariate Approaches for Process Model Forecasting},
  author={Yu, Yongbo and Peeperkorn, Jari and De Smedt, Johannes and De Weerdt, Jochen},
  booktitle={International Conference on Process Mining},
  pages={279--292},
  year={2024},
  organization={Springer}
}
```

