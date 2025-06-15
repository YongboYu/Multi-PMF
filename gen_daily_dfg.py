import os
import json
import numpy as np
from datetime import datetime, timedelta

from lib.DFs_struction import read_data_daily, determine_trim_percentage, ActivityPair
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer



def create_act_map_and_dfg(dataset, input_path, output_map_path, output_dfg_path):
    """
    Create the activity map for the given dataset
    :param dataset: the name of the dataset
    :param input_path: the path to the input dataset
    :param output_path: the path to the output dataset
    :return:
    """
    input_file = os.path.join(input_path, f'{dataset}.xes')

    # import the log
    variant = xes_importer.Variants.ITERPARSE
    paras = {variant.value.Parameters.MAX_TRACES: 1000000000}
    log = xes_importer.apply(input_file, parameters=paras)

    # get the start and end time
    timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
    start_time = min(timestamps)  
    end_time = max(timestamps)  

    # get activity names
    activity_names = pm4py.get_event_attribute_values(log, 'concept:name')
    no_act = len(activity_names)  
    act_map = {}
    reverse_map = {}
    for a, value in enumerate(activity_names.keys()):
        act_map[value] = a
        reverse_map[a] = value

    # store all directly-follows occurrences as activity pairs
    apairs = []
    for t, trace in enumerate(log):
        for e, event in enumerate(trace):
            if e == len(trace) - 1:
                continue
            ap = ActivityPair(event['concept:name'], trace[e + 1]['concept:name'], trace[e + 1]['time:timestamp'],
                              event,
                              trace[e + 1], t)
            apairs.append(ap)

    sorted_aps = sorted(apairs)
    print('#DFs:', len(sorted_aps))  

    no_daily_intervals_all = (end_time - start_time).days + 1  
    interval_width = int(len(sorted_aps) / no_daily_intervals_all)
    print('interval width: ', interval_width) 

    # save the activity maps as JSON for adjacent matrix
    output_act_map = os.path.join(output_map_path, f'{dataset}_act_map.json')
    output_reverse_map = os.path.join(output_map_path, f'{dataset}_reverse_map.json')
    with open(output_act_map, 'w') as f:
        json.dump(act_map, f)
    with open(output_reverse_map, 'w') as f:
        json.dump(reverse_map, f)

    print(f'Activity map for {dataset} is created.')

    # create the DFG matrix
    output_dfg_file= os.path.join(output_dfg_path, f'daily_dfg_mat_{dataset}.npy')
    if os.path.isfile(output_dfg_file):
        with open(output_dfg_file, 'rb') as f:
            dfg_time_matrix_org = np.load(f)
        print('Read matrix:', dfg_time_matrix_org.shape)
    else:
        print('Creating DFG matrix')
        dfg_time_matrix_org, interval_timings = read_data_daily(no_daily_intervals_all, sorted_aps, act_map, log,
                                                                dataset)
        with open(output_dfg_file, 'wb') as f:
            np.save(f, dfg_time_matrix_org)

    print(f'DFG matrix for {dataset} is created.')

input_path = './event_logs/processed'
output_act_path = './data/act_map'
output_dfg_path = './data/daily_dfg_mat'
datasets = ['BPI2019_1', 'Hospital_Billing', 'RTFMP']

for dataset in datasets:
    create_act_map_and_dfg(dataset, input_path, output_act_path, output_dfg_path)
