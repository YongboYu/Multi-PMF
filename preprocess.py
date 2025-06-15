import os
from datetime import datetime, timedelta

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer


def preprocess_event_log(dataset, input_path, output_path, filter_perc=0.0001, trim_p=0.1):
    """
    Preprocess the event log by filtering the infrequent variants and trimming the log
    :param dataset: the name of the dataset
    :param input_path: the path of the input event log
    :param output_path: the path of the output event log
    :param filter_perc: the percentage of the infrequent variants to be filtered
    :param trim_p: the percentage of the log to be trimmed
    :return: None
    """
    # file paths
    input_file = os.path.join(input_path, f'{dataset}.xes')
    output_file = os.path.join(output_path, f'{dataset}.xes')
    
    os.makedirs(output_path, exist_ok=True)

    # import the log
    variant = xes_importer.Variants.ITERPARSE
    paras = {variant.value.Parameters.MAX_TRACES: 1000000000}
    org_log = xes_importer.apply(input_file, parameters=paras)

    # filter the infrequent variants
    filter_log = pm4py.filter_variants_by_coverage_percentage(org_log, filter_perc)

    # add artificial start and end activities to traces
    log = pm4py.insert_artificial_start_end(filter_log, activity_key='concept:name',
                                            case_id_key='case:concept:name', timestamp_key='time:timestamp')

    # get the start and end time
    timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
    start_time = min(timestamps)
    end_time = max(timestamps)

    # trim the event log to 80%
    trim_days = int((end_time - start_time).days * trim_p)
    start_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    trim_start = (start_date + timedelta(days=trim_days)).strftime('%Y-%m-%d %H:%M:%S')
    trim_end = (end_date - timedelta(days=trim_days)).strftime('%Y-%m-%d %H:%M:%S')
    trimmed_filter_log = pm4py.filter_time_range(log, trim_start, trim_end)

    # save the filtered log
    pm4py.write_xes(trimmed_filter_log, output_file)

    # save the information to a text file
    with open('./event_logs/processed/log_info_preprocess.txt', 'a') as f:
        f.write(f'Dataset: {dataset}\n')
        f.write(f'Number of traces (Original Log): {len(org_log)}\n')
        f.write(f'Start time (org): {start_time}\n')
        f.write(f'End time (org): {end_time}\n')
        f.write(f'Filter percentage: {filter_perc} + {filter_perc}\n')
        f.write(f'Number of traces (Filtered Log): {len(filter_log)} ({round(filter_log/org_log*100, 2)}%)\n')
        f.write(f'Trim percentage: {trim_p}\n')
        f.write(f'Start time: {trim_start}\n')
        f.write(f'End time: {trim_end}\n')
        f.write(f'Number of traces (Trimmed Log): {len(trimmed_filter_log)} ({round(filter_log/org_log*100, 2)}%)\n\n')

    print(f'Preprocessing of {dataset} is done!')

    return trimmed_filter_log



input_path = './event_logs/original'
output_path = './event_logs/processed'
datasets = ['BPI2019_1', 'Hospital_Billing', 'RTFMP']

for data in datasets:
    preprocess_event_log(data, input_path, output_path)






