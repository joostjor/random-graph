import copy
import json
from pathlib import Path
import tarfile 
import tempfile 
import time

import numpy as np
from scipy.sparse import find

from .history import SpreadingStateHistory


class SpreadingModel:
    """
    This is a generic class which simulates an epidemic model on a given SpatialGraph, using a SpreadingState to
    store information about specific vertices. It handles doing multiple runs, recording the results (using the
    SpreadingStateHistory class), and advancing time within each individual run. The specific epidemic model will
    then inherit from this class (e.g. with the SEIRAppModel class), overriding the infection_state_maker,
    determine_infections and infection_generator methods to determine its specific dynamics.
    """
    def __init__(self):
        self._parameters_base = []
        self._parameters = []

    # Names of the numerical parameters of the model, e.g. the I -> R transition rate in SIR.
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = self._parameters_base + parameters

    @property
    def parameter_dict(self):
        return dict([(param, getattr(self, param))
                     for param in self.parameters])

    def describe(self):
        summary_dict = copy.deepcopy(self.parameter_dict)
        summary_dict["time_generated"] = int(time.time())
        summary_dict["class"] = type(self).__name__
        return summary_dict

    # The class of spreading states used in the model, e.g. SIRSpreadingState.
    @property
    def spreading_state_class(self):
        raise NotImplementedError("spreading_state_class not implemented.")

    # Returns a function to create the initial state of the model. Also sets the center_vertex of that spreading
    # state to be the vertex any animations should be centered on, typically the first infected vertex.
    def infection_state_maker(self, number_of_vertices, number_of_infections=1):
        raise NotImplementedError("Infection state maker not implemented.")

    # Use this for infection_state_maker if you just want some uniformly random infected vertices.
    def default_infection_state_maker(self, number_of_vertices, number_of_infections=1):
        def maker():
            infection = self.infection_generator(number_of_infections)
            vertex_indices = np.random.choice(np.arange(number_of_vertices),
                                              size=number_of_infections,
                                              replace=False)
            initial_state = self.spreading_state_class(number_of_vertices)
            initial_state.infect_vertices(vertex_indices, infection)
            initial_state.center_vertex = vertex_indices[0]
            return initial_state
        return maker

    # Given a list of (possibly weighted) edges between infected vertices and susceptible vertices, output the
    # sublist of vertices to be infected that time step.
    def determine_infections(self, distances_from_infection):
        raise NotImplementedError("Infection determiner not implemented")

    # Generates the specified number of infection trajectories, to be passed to SpreadingState.infect_vertices().
    def infection_generator(self, number_of_infections):
        raise NotImplementedError("Infection generator not implemented")

    # Runs the given spreading state for a single round on the graph with the given distance matrix, modifying it in
    # place. Note that the spreading state is NOT a part of the model class - we can pass a single spreading state
    # between multiple models if we want.
    def _single_round(self,
                      spreading_state,
                      distance_matrix):
        susceptible_vertices = spreading_state.susceptible_vertices
        infected_vertices = spreading_state.infectious_vertices
        distance_from_infected = distance_matrix[infected_vertices].T[susceptible_vertices]

        nonzero_indices_rows, _, distances = find(distance_from_infected)

        infection_booleans = self.determine_infections(distances)
        idx_to_infect = np.unique(nonzero_indices_rows[infection_booleans])
        vertices_to_infect = susceptible_vertices[idx_to_infect]
        spreading_state.increase_time()
        spreading_state.infect_vertices(vertices_to_infect,
                                        self.infection_generator(len(vertices_to_infect)))

    # Completes a single run of the model, using the given graph_determiner function to determine which underlying
    # graph is used at each time step, and rerunning the model (using the given initial state maker) if we end without
    # infecting enough vertices as determined by min_infected. Note that in the current paper, we will always use the
    # same graph at all time steps.
    def single_run(self,
                   make_initial_state,
                   graph_determiner,
                   max_rounds,
                   min_infected=0):
        successful_run = False
        history_list = None
        new_state = None
        while not successful_run:
            new_state = make_initial_state()
            history_list = [new_state.state_list]
            graph = None

            for t in range(max_rounds):
                if not successful_run:
                    if len(new_state.susceptible_vertices) <= new_state.nr_of_vertices - min_infected:
                        successful_run = True
                if not new_state.state_active:
                    break

                graph = graph_determiner(t, new_state, graph)
                self._single_round(new_state, graph.distance_matrix)
                history_list = np.vstack((history_list, new_state.state_list))

        return SpreadingStateHistory(STATES=new_state.STATES,
                                     STATE_LABELS=new_state.STATE_LABELS,
                                     history_list=history_list,
                                     center_vertex=new_state.center_vertex)

    # Runs the model on the given vertex set and graph_determiner. Writes to file if one is specified, including a
    # brief description of the model if one is specified. Save_full_hist is a bit slower, but saves the entire
    # history of the model - otherwise only a few summary statistics are saved.
    def run_model(self,
                  vertex_set,
                  graph_determiner,
                  initial_state=None,
                  nr_of_initial_infections=1,
                  max_rounds=1000,
                  nr_of_runs=1,
                  min_infected=0,
                  write_to_file=None,
                  description=None,
                  description_func=None,
                  save_full_hist=False,
                  return_results=False,
                  output_progress=True):
        """
        :params
        - vertex_set: VertexSet on which the spreading model will take place
        - graph_determiner: function that determines per time step what graph to consider
        - initial_state: SpreadingState 
        - nr_of_initial_infections: Number of initially infected vertices.
        - max_rounds: max number of time steps
        - nr_of_runs: number of independent simulation runs
        - min_infected: if the epidemic ends before this many people are infected, re-run the model.
        - write_to_file: either None or a path if results need to be stored
        - description: json serializable object that is stored with the results
        - save_full_hist: if true, saves all the transitions of the vertices, 
                          if false: stores the number of vertices in every state per time step
        - output_progress: if true, prints a line to the console on completion of each run (so we can see how fast
                           it's going).
        """
        make_initial_state = (self.infection_state_maker(vertex_set.number_of_vertices,
                                                         nr_of_initial_infections)
                              or (lambda _: initial_state.copy))        
        def make_description_dict(description, description_func):
            if description is None and description_func is None:
                return
            descr = description or {}
            if description_func is not None:
                descr = {**descr, 
                            **description_func()}
            return descr  
        
        if write_to_file is None:
            results = []
            for i in range(nr_of_runs):
                results.append(self.single_run(make_initial_state,
                                               graph_determiner,
                                               max_rounds,
                                               min_infected))
                if output_progress and (i % 5 == 4):
                    print("Run", i+1, "of", nr_of_runs, "complete.")
            description = make_description_dict(description,
                                                description_func)
            if nr_of_runs == 1:
                return results[0], description
            else:
                return results, description
        else:
            number_length = len(str(nr_of_runs))
            if not str(write_to_file).endswith(".tar.gz"):
                filename_full_hist = Path(str(write_to_file) + ".tar.gz")
            else:
                filename_full_hist = Path(str(write_to_file))            
            filename_summary = filename_full_hist.parent / ("summaries_" + filename_full_hist.name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                with tarfile.open(filename_full_hist, "w:gz") as tar_full_hist:
                    with tarfile.open(filename_summary, "w:gz") as tar_summary:                       
                        if save_full_hist:
                            vertex_set.save(tmp_path / "vertex_set.npz")
                            tar_full_hist.add(tmp_path / "vertex_set.npz", arcname="vertex_set.npz")
                        if return_results:
                            results = []
                        try:
                            for i in range(nr_of_runs):
                                fn = "run_" + str(i).zfill(number_length)
                                result = self.single_run(make_initial_state,
                                                         graph_determiner,
                                                         max_rounds,
                                                         min_infected)
                                if return_results:
                                    results.append(result)
                                if save_full_hist:
                                    result.save((tmp_path / (fn + ".tar.gz")))
                                    tar_full_hist.add(tmp_path / (fn + ".tar.gz"), arcname=(fn + ".tar.gz"))
                                result.save_summary((tmp_path / fn))
                                tar_summary.add(tmp_path / (fn + ".npz"), arcname=(fn + ".npz"))                            
                                if output_progress and (i % 5 == 4):
                                    print("Run", i+1, "of", nr_of_runs, "complete.")
                        finally:                          
                            description = make_description_dict(description,
                                                                description_func)
                            if description is not None:                            
                                
                                with open(tmp_path / "description.json", "w") as f:
                                    json.dump(make_description_dict(description,
                                                                    description_func), 
                                            f, indent=4, sort_keys=True)
                                tar_full_hist.add(tmp_path / "description.json", arcname="description.json")
                                tar_summary.add(tmp_path / "description.json", arcname="description.json")
            if not save_full_hist:
                filename_full_hist.unlink()
            if return_results:
                return results, description
