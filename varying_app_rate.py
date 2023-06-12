import argparse
from collections import namedtuple
import json
from pathlib import Path
import sys
import time

import numpy as np

from graph_lib.spatial_graph import VertexSet, SpatialGraph, ConfigurationModel
from spreading_lib.SEIRApp_model import SEIRAppModel, SEIRAppStates


if __name__ == "__main__":
    # Hardcoded model parameters --- change as needed.
    time_to_infectious_mean = 3      # Average time to transition E -> I
    time_to_symptoms_mean = 2.5      # Average time to transition I -> Sy when appropriate
    time_to_removal_mean = 7         # Average time to transition I -> R (possibly by way of Sy)
    quarantine_len = 14              # Quarantine length in time steps
    runs_per_setting = 1             # Number of runs for each value of the binary search for the critical app-rate.
    
    max_rounds=5000
    initial_infections=100
    
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_input", type=str,
                        help="path to graph")
    parser.add_argument("beta", type=float, 
                        help="probability of infecting a neighbor")
    parser.add_argument("symptom_prob", type=float, 
                        help="probability of showing symptoms and to quarantine if infected")
    parser.add_argument("app_rate", type=float,
                        help="probability of using app and to notify if showing symptoms")
    parser.add_argument("method", type=str,
                        help="random/degree/recommender")
    parser.add_argument("filename_output", type=str,
                        help="path to store tar.gz with summaries")
    args = parser.parse_args()
    #if args.method is not "random" and args.app_rate in [0, 1]:
    #    print(f"Exit: {args.method}, {args.app_rate}") 
    #    sys.exit()
    
    filepath_output = Path(args.filename_output)
    """
    with open("processed_varying_app_rate.txt") as f:
        processed_files = [l.split("#")[0] for l in f.readlines()]
    processed_files = ["/".join(f.split("/")[-3:]) for f in processed_files]
    if f"{filepath_output.parent.parent.name}/{filepath_output.parent.name}/summaries_{filepath_output.name.split('#')[0]}" in processed_files:
        print(f"Already processed {filepath_output.parent.parent.name}/{filepath_output.parent.name}/summaries_{filepath_output.name.split('#')[0]}")
        sys.exit()
    if filepath_output.name.startswith("run"):
        old_name = filepath_output.name[6:]
        if f"{filepath_output.parent.parent.name}/{filepath_output.parent.name}/summaries_{old_name.split('#')[0]}" in processed_files:
            print(f"Already processed {filepath_output.parent.parent.name}/{filepath_output.parent.name}/summaries_{old_name.split('#')[0]}")
            sys.exit()
    """
    print(f"Process {filepath_output}")
    if not Path(args.graph_input).name.startswith("cm"):
        graph = SpatialGraph.load_graph(args.graph_input)    
    else:
        graph = ConfigurationModel.load_graph(args.graph_input)
    graph_determiner = lambda *args, **kwargs: graph

    spreading_model = SEIRAppModel(infection_prob=args.beta,
                                   infection_rate=1/time_to_infectious_mean,
                                   removal_rate=1/time_to_removal_mean,
                                   method=args.method,
                                   graph=graph,
                                   app_rate=args.app_rate,
                                   symptom_prob=args.symptom_prob,
                                   symptom_rate=1/time_to_symptoms_mean,
                                   quarantine_len=quarantine_len)
    description = graph.describe()
    description_func = spreading_model.describe
    
    spreading_model.run_model(vertex_set=graph.vertex_set,
                              graph_determiner=graph_determiner,
                              nr_of_initial_infections=initial_infections,
                              nr_of_runs=runs_per_setting,
                              max_rounds=max_rounds,
                              write_to_file=filepath_output,
                              description=description,
                              description_func=description_func,
                              return_results=True)    
    
