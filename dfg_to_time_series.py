import numpy as np
import pandas as pd

def flat_dfg_matrix(dfg_time_matrix):
    days, activities, _ = dfg_time_matrix.shape
    return dfg_time_matrix.reshape(days, activities * activities)

def process_dataset(dataset):
    daily_dfg = np.load(f'./data/daily_dfg_mat/daily_dfg_mat_{dataset}.npy')
    daily_dfg_flat = flat_dfg_matrix(daily_dfg)
    data = pd.DataFrame(daily_dfg_flat)

    if dataset == 'BPI2019_1':
        date_index = pd.date_range(start='2018-02-08', periods=data.shape[0])
    elif dataset == 'Hospital_Billing':
        date_index = pd.date_range(start='2013-04-04', periods=data.shape[0])
    elif dataset == 'RTFMP':
        date_index = pd.date_range(start='2001-05-04', periods=data.shape[0])

    data.index = date_index
    data = data.loc[:, (data != 0).any(axis=0)]
    data.to_hdf(f'./dataset/{dataset}.h5', key='df')

datasets = ['BPI2019_1', 'Hospital_Billing', 'RTFMP']
for dataset in datasets:
    process_dataset(dataset)
