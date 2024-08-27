import numpy as np
import pm4py
import pytz
import os
from datetime import datetime, timedelta

from pm4py.algo.discovery.dfg.variants import native as dfg_factory
from collections import Counter
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.algo.discovery.dfg.variants import native

from pm4py.objects.dfg.utils.dfg_utils import infer_start_activities, infer_end_activities
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes.variants import etree_xes_exp as exporter


class ActivityPair:

    def __init__(self, a1, a2, timestamp, event, event2, trace_no):
        self.a1 = a1
        self.a2 = a2
        self.timestamp = timestamp
        self.event = event
        self.event2 = event2
        self.trace_no = trace_no

    def __str__(self):
        return self.a1 + ' before ' + self.a2 + ' at ' + str(self.timestamp)

    def __gt__(self, other):
        if self.timestamp > other.timestamp:
            return True
        else:
            return False






def read_data_daily(no_intervals, sorted_aps, act_map, log, dataset):
    timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
    print('Earliest:', min(timestamps))
    print('Latest:', max(timestamps))
    start_date = min(timestamps).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = max(timestamps).replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
    interval_length = (end_date - start_date) / no_intervals
    print('Interval length:', interval_length)

    no_act = len(act_map.keys())

    dfg_time_matrix = np.zeros([no_intervals, no_act, no_act], dtype=int)

    interval_timing = []
    no_events_sums = 0
    no_events_logs = 0
    no_dfs = 0
    for i in range(0, no_intervals):
        print('Interval ', i, '/', no_intervals)
        lower_bound = start_date + i * interval_length
        if i == (no_intervals - 1):
            upper_bound = start_date + (i + 1) * interval_length * 2
        else:
            upper_bound = start_date + (i + 1) * interval_length
        lb = lower_bound
        ub = upper_bound
        print(lb)
        print(ub)

        dfs = []
        empty_mat = np.zeros([no_act, no_act], dtype=float)

        filtered_events = {}
        # start = Event()
        # end = Event()
        # start['concept:name'] = str(act_map['start'])
        # end['concept:name'] = str(act_map['end'])
        highest = datetime(1970, 1, 1, tzinfo=pytz.UTC)
        lowest = datetime(2050, 1, 1, tzinfo=pytz.UTC)

        count = 0
        for df in sorted_aps:
            if ub > df.event2['time:timestamp'] >= lb:# and ub > df.event['time:timestamp'] >= lb:
                dfs.append(df)

        no_dfs += len(dfs)

        log_dfs = {}
        for df in dfs:
            if df.trace_no not in log_dfs.keys():
                log_dfs[df.trace_no] = []
            log_dfs[df.trace_no].append(df)

        for trace_no, dfss in log_dfs.items():
            # print('\nTrace:', trace_no)
            sorted_dfs = sorted(dfss)
            filtered_events[trace_no] = []
            for df in sorted_dfs:
                # print(df)
                filtered_events[trace_no].append(df.event)
                no_events_sums += 1
            filtered_events[trace_no].append(sorted_dfs[len(sorted_dfs)-1].event2)
            no_events_sums += 1

        print('#traces:', len(log_dfs))

        # for trace_no, events in filtered_events.items():
        #     empty_mat[act_map['start'], act_map[events[0]['concept:name']]] += 1
        #     empty_mat[act_map[events[-1]['concept:name']], act_map['end']] += 1

        # Export filtered events to interval event logs
        new_log = EventLog()
        no_eve = 0
        for t, trace in enumerate(log):
            new_trace = Trace()
            # new_trace.append(start)
            for trace_no, events in filtered_events.items():
                if t == trace_no:
                    for event in trace:
                        if event in events:
                            if event['time:timestamp'] < lowest:
                                lowest = event['time:timestamp']
                            if event['time:timestamp'] > highest:
                                highest = event['time:timestamp']
                            new_event = Event()
                            new_event['concept:name'] = str(act_map[event['concept:name']])
                            new_trace.append(new_event)
                            no_events_sums += 1
                            no_eve += 1
            if len(new_trace) > 0:
                # new_trace.append(end)
                new_log.append(new_trace)
        os.makedirs(f'./data/daily_log/{dataset}', exist_ok=True)
        exporter.apply(new_log, f'./data/daily_log/{dataset}/' + dataset + '_log_interval_' + str(i) + '-'
                       + str(no_intervals) + '_daily.xes')

        # print('no eve:', no_eve)
        for act_pair in dfs:
            a1 = act_map[act_pair.a1]
            a2 = act_map[act_pair.a2]
            empty_mat[a1, a2] += 1

        dfg_time_matrix[i] = empty_mat
        interval_timing.append((lowest, highest))
    print('Event sums:', no_events_sums)
    print('Event logs:', no_events_logs)
    print('#DFS:', no_dfs)

    return dfg_time_matrix, interval_timing



def determine_trim_percentage(dfg_time_matrix, trim_p):
    all_intervals = dfg_time_matrix.shape[0]
    trim_days = int(all_intervals * trim_p)
    dfg_time_matrix_trimmed = dfg_time_matrix[trim_days:(all_intervals - trim_days), ::, ::]

    return dfg_time_matrix_trimmed



def get_dfg_matrix(sorted_aps, act_map):

    no_act = len(act_map.keys())

    dfg_time_matrix = np.zeros([no_act, no_act], dtype=int)

    for act_pair in sorted_aps:
        a1 = act_map[act_pair.a1]
        a2 = act_map[act_pair.a2]
        dfg_time_matrix[a1, a2] += 1

    return dfg_time_matrix