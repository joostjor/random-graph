import argparse
from collections import namedtuple
import json
from pathlib import Path
import sys
import time

import numpy as np

from graph_lib.spatial_graph import VertexSet, GIRG
from spreading_lib.history import SpreadingStateHistory
from spreading_lib.SEIRApp_model import SEIRAppModel, SEIRAppStates


OutbreakInfo = namedtuple("OutbreakInfo", ["height", "volume", "time"])
outbreakinfo_to_dict = lambda info: {k: int(v) for k, v in info._asdict().items()}

def minimal_app_compliance(graph, 
                           beta, 
                           infectious_rate, 
                           symptom_rate, 
                           symptom_prob,
                           removal_rate,
                           quarantine_len,
                           proportion_infected_max=0.05,
                           proportion_successful_runs_min=.9,
                           runs_per_setting=5,
                           max_rounds=5000,
                           initial_infections=100,
                           tolerance=0.005, 
                           peak_dict=None, 
                           write_to_file=None):
    if write_to_file is None:
        write_to_file = lambda app_rate: None
    graph_determiner = lambda *args, **kwargs: graph
    app_rate = 0
    app_rate_diff = 1
    if peak_dict is None:
        peak_dict = dict()
    while 2*app_rate_diff > tolerance:
        spreading_model = SEIRAppModel(infection_prob=beta,
                                       infection_rate=infectious_rate,
                                       removal_rate=removal_rate,
                                       app_rate=app_rate,
                                       symptom_prob=symptom_prob,
                                       symptom_rate=symptom_rate,
                                       quarantine_len=quarantine_len)
        description = {**spreading_model.describe(),
                       **graph.describe()}

        if runs_per_setting == 1:
            results = [results]

        filepath_full_hist = Path(write_to_file(app_rate))
        filename_summary_pattern = f"summaries_{filepath_full_hist.name.split('#')[0]}*"
        similar_files = list(filepath_full_hist.parent.glob(filename_summary_pattern))
        
        reading_succeeded = False
        if len(similar_files) > 0:
            try:
                print(f"for app_rate={app_rate} found {similar_files[0]}")
                nr_dicts = [r["nr_dict"] for r in SpreadingStateHistory.load_bundled_summaries(similar_files[0]).values()]
                reading_succeeded = True
            except:
                pass
        if not reading_succeeded:
            print(f"simulate app_rate={app_rate}")
            results = spreading_model.run_model(vertex_set=graph.vertex_set,
                                                graph_determiner=graph_determiner,
                                                nr_of_initial_infections=initial_infections,
                                                nr_of_runs=runs_per_setting,
                                                write_to_file=write_to_file(app_rate),
                                                description=description,
                                                return_results=True)
            nr_dicts = [r.nr_over_time() for r in results]
        



        outbreak_volumes = np.array([nr_dict.get(SEIRAppStates.REMOVED)[-1]
                                     + nr_dict.get(SEIRAppStates.Q_REMOVED)[-1]
                                     for nr_dict in nr_dicts])
        
        total_infected_arrays = [nr_dict.get(SEIRAppStates.INFECTED) 
                                 + nr_dict.get(SEIRAppStates.Q_INFECTED)
                                 + nr_dict.get(SEIRAppStates.EXPOSED)
                                 + nr_dict.get(SEIRAppStates.Q_EXPOSED)
                                 + nr_dict.get(SEIRAppStates.SYMPTOMATIC)
                                 for nr_dict in nr_dicts]
        peak_times = [np.argmax(array) for array in total_infected_arrays]
        
        peak_dict[app_rate] = [OutbreakInfo(volume=volume, 
                                            height=infected_array[peak_time], 
                                            time=peak_time) 
                               for (volume, infected_array, peak_time) in zip(outbreak_volumes, 
                                                                              total_infected_arrays, 
                                                                              peak_times)]
        
        successful_runs = outbreak_volumes < (graph.nr_of_nodes * proportion_infected_max)       
        if np.mean(successful_runs) > proportion_successful_runs_min:
            if app_rate == 0:
                return app_rate, peak_dict
            app_rate -= app_rate_diff
        elif app_rate == 1:
            return app_rate, peak_dict
        else:
            app_rate += app_rate_diff
        app_rate_diff /= 2
    return app_rate, peak_dict

if __name__ == "__main__":
    # Hardcoded model parameters --- change as needed.
    time_to_infectious_mean = 3      # Average time to transition E -> I
    time_to_symptoms_mean = 2.5      # Average time to transition I -> Sy when appropriate
    time_to_removal_mean = 7         # Average time to transition I -> R (possibly by way of Sy)
    quarantine_len = 14              # Quarantine length in time steps
    proportion_infected_max = 0.05   # Maximum ``acceptable'' number of removed vertices at end of outbreak
    proportion_successful_runs = .9  # To be deprecated?
    runs_per_setting = 10            # Number of runs for each value of the binary search for the critical app-rate.
    tolerance = 0.005                # Allowed error in critical app-rate.

    parser = argparse.ArgumentParser()
    parser.add_argument("graph_input", type=str,
                        help="path to graph")
    parser.add_argument("beta", type=float, 
                        help="probability of infecting a neighbor")
    parser.add_argument("symptom_prob", type=float, 
                        help="probability of showing symptoms and to quarantine if infected")
    parser.add_argument("filename_output", type=str,
                        help="path to store json-file with results")
    args = parser.parse_args()

    graph = GIRG.load_graph(args.graph_input)
    filepath_output = Path(args.filename_output)
    if len(list(filepath_output.parent.glob("beta={0:.2f}_symptom_prob={1:.1f}*.json".format(args.beta, args.symptom_prob)))) > 0:
        print(f"{filepath_output} exists")
        sys.exit()
    summary_filename = lambda app_rate: filepath_output.parent / f"app_rate={app_rate}_{filepath_output.name[:-5]}.tar.gz"
    app_rate, peak_dict = minimal_app_compliance(graph=graph,
                                                 beta=args.beta,
                                                 infectious_rate=1/time_to_infectious_mean,
                                                 symptom_rate=1/time_to_symptoms_mean,
                                                 symptom_prob=args.symptom_prob,
                                                 removal_rate=1/time_to_removal_mean,
                                                 runs_per_setting=runs_per_setting,
                                                 quarantine_len=quarantine_len,
                                                 tolerance=tolerance,
                                                 write_to_file=summary_filename)
               
    peak_dict_serialized = {str(app_rate): [outbreakinfo_to_dict(info) for info in infos] 
                            for app_rate, infos in peak_dict.items()}

    with open(filepath_output, 'w') as f:
        json.dump({"results": peak_dict_serialized,
                   "optimal_rate": app_rate,
                   "proportion_infected_max": proportion_infected_max, 
                   "tolerance": tolerance,
                   "time_to_infectious_mean": time_to_infectious_mean,
                   "time_to_symptoms_mean": time_to_symptoms_mean,
                   "time_to_removal_mean": time_to_removal_mean,
                   "quarantine_len": quarantine_len,
                   "proportion_successful_runs": proportion_successful_runs,
                   "beta": args.beta,
                   "runs_per_setting": runs_per_setting,
                   "symptom_prob": args.symptom_prob},
                   f, indent=4, sort_keys=True)
