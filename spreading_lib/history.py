import copy
from functools import reduce
import json
from pathlib import Path
import tarfile
import tempfile 

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz


class SpreadingStateHistory:
    """This class contains the history of an entire run of a SpreadingModel as an array history_list of state vectors,
    with methods for saving, loading, and outputting various plots and statistics. history_list[t,v] is the (numerical)
    state of vertex v at time t. For efficiency, we only actually save and load the changes between states at each step
    (the "switch matrix"). This lets us use numpy's sparse matrix handling routines and makes things more efficient in
    models where not many vertices change state at each step. Note that this class does *not* contain the vertex set
    (which will typically be in its own file to be shared between runs)."""
    def __init__(self,
                 STATES,
                 STATE_LABELS,
                 history_list,
                 center_vertex,
                 graph_description=None,
                 spreading_description=None):
        self._STATES = STATES
        self._STATE_LABELS = STATE_LABELS
        self.center_vertex = center_vertex
        self.history_list = history_list
        self.graph_description = graph_description
        self.spreading_description = spreading_description

    def __len__(self):
        return len(self.history_list)

    @property
    def STATES(self):
        return self._STATES

    @property
    def STATE_LABELS(self):
        return self._STATE_LABELS

    # Load a history from the "switch matrix", which contains all the state changes at each time.
    @classmethod
    def from_switch_matrix(cls, STATES, STATE_LABELS, center_vertex, switch_matrix, graph_description=None,
                           spreading_description=None):

        return cls(STATES=STATES,
                   STATE_LABELS=STATE_LABELS,
                   history_list=cls.load_switch_matrix(switch_matrix),
                   center_vertex=center_vertex,
                   graph_description=graph_description,
                   spreading_description=spreading_description)

    # Retrieve the switch matrix for the run currently stored.
    @property
    def switch_matrix(self):
        return np.vstack([self.history_list[0, :], np.diff(self.history_list, axis=0)])

    @classmethod
    def load_switch_matrix(cls, switch_matrix):
        return switch_matrix.toarray().cumsum(axis=0)

    # Loads a run saved by the save method below into the current class.
    @classmethod
    def load_from_file(cls, filename):
        with tarfile.open(filename, "r:gz") as tar:
            switch_matrix = load_npz(tar.extractfile("switch_matrix.npz"))
            STATES, STATE_LABELS, center_vertex = np.load(tar.extractfile("state_data.npy"), allow_pickle=True)
            description = json.load(tar.extractfile("description.json"))
        return cls.from_switch_matrix(STATES=STATES, 
                                      STATE_LABELS=STATE_LABELS, 
                                      center_vertex=center_vertex,
                                      switch_matrix=switch_matrix, 
                                      **description)

    def describe(self, omit_time_spreading=False):
        graph_description = copy.deepcopy(self.graph_description)
        spreading_description = copy.deepcopy(self.spreading_description)
        if omit_time_spreading:
            spreading_description.pop("time_generated", None)
        return {"graph_description": graph_description,
                "spreading_description": spreading_description}

    # Saves the entire run, omitting only debug information.
    def save(self, filename):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "switch_matrix.npz", csr_matrix(self.switch_matrix))
            with open(tmp_path / "description.json", "w") as f:
                json.dump(self.describe(), f, indent=4, sort_keys=True)
            np.save(tmp_path / "state_data.npy", (self.STATES, self.STATE_LABELS, self.center_vertex), allow_pickle=True)
            with tarfile.open(filename, "w:gz") as tar:
                tar.add(tmp_path / "switch_matrix.npz", arcname="switch_matrix.npz")
                tar.add(tmp_path / "state_data.npy", arcname="state_data.npy")
                tar.add(tmp_path / "description.json", arcname="description.json")

    # Saves only a few key statistics (number of total/new vertices in each state, first time at which each vertex
    # enters each state).
    def save_summary(self, filename):
        nr_dict = {state: self.nr_over_time(state) for state in self.STATES}
        new_dict = {state: self.nr_new_over_time(state) for state in self.STATES}
        first_dict = {state: self.first_time(state) for state in self.STATES}

        np.savez_compressed(filename, 
                            nr_dict=nr_dict, 
                            new_dict=new_dict, 
                            first_dict=first_dict, 
                            center_vertex=self.center_vertex)

    # Load and return the key statistics from a summary saved by save_summary.
    @staticmethod
    def load_summary(filename):
        loaded_file = np.load(filename, allow_pickle=True)
        summary = {k: v for k, v in loaded_file.iteritems()}
        return {k: v.item() for k, v in summary.items()}

    # Load a collection of summaries from a tarball.
    @staticmethod
    def load_bundled_summaries(filename, min_length=1, description_params=None):
        if description_params is None:
            description_params = []        
        with tarfile.open(filename) as tar:    
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tar.extractall(tmp_path)
                summaries = {run.name: SpreadingStateHistory.load_summary(run)
                             for run in tmp_path.glob("run_*")}
                if len(description_params) > 0:
                    with open(tmp_path / "description.json", "r") as f:
                        description = json.load(f)

        def summary_length(summary):
            return len(list(summary["nr_dict"].values())[0])
        if min_length > 1:            
            summaries = dict([(run, summary) for run, summary in summaries.items() 
                              if summary_length(summary) >= min_length])
        if len(description_params) > 0:
            return summaries, {param: value for (param, value) in description.items() 
                               if param in description_params}
        return summaries

    # Turns a summary into a PANDAS DataFrame.
    @staticmethod
    def summary_to_time_df(summary, run_id=None):
        df_tmp1 = pd.DataFrame(summary["nr_dict"])
        df_tmp1 = df_tmp1.rename(columns={k.value: k.name.capitalize() 
                                          for k in summary["nr_dict"].keys()})
        df_tmp2 = pd.DataFrame(summary["new_dict"])
        df_tmp2 = df_tmp2.rename(columns={k.value: f"Newly {k.name.capitalize()}" 
                                          for k in summary["new_dict"].keys()})
        df_tmp = df_tmp1.join(df_tmp2)
        if run_id is not None:
            df_tmp["Run_ID"] = run_id
        df_tmp = df_tmp.reset_index()
        df_tmp = df_tmp.rename(columns={"index": "Time"})    
        return df_tmp

    # Loads a collection of summaries from a tarball to a PANDAS DataFrame.
    @staticmethod
    def load_bundled_summaries_to_time_df(filename, bundle_id=None, min_length=1, description_params=None):
        if description_params is None:
            summaries = SpreadingStateHistory.load_bundled_summaries(filename, 
                                                                     min_length=min_length)
        else:
            summaries, description = SpreadingStateHistory.load_bundled_summaries(filename, 
                                                                                  min_length=min_length,
                                                                                  description_params=description_params)
        df = pd.concat([SpreadingStateHistory.summary_to_time_df(summary, run_id=run_id) 
                        for run_id, summary in summaries.items()])
        if type(description_params) is list and len(description_params) > 0:
            for k, v in description.items():
                if k == "app_coverages":
                    df[k] = np.mean(v) 
                else:
                    df[k] = v
        if bundle_id is not None:
            df["Bundle_ID"] = bundle_id
        return df

    @property
    def length(self):
        return len(self.history_list)

    @property
    def nr_vertices(self):
        return len(self.history_list[0])

    # Returns the list of vertices in the given state at the given time
    def state_filter(self, state_no, time):
        return np.where(self.history_list[time, :] == state_no)

    # Returns a list of the number of total vertices in the given state at each time step.
    def nr_over_time(self, state_no=None):
        dictionaries = [dict(zip(*np.unique(state, return_counts=True)))
                        for state in self.history_list]

        def append(dir_over_time, dir_new):
            for s in self.STATES:
                dir_over_time[s].append(dir_new.get(int(s), 0))
            return dir_over_time
        d_over_time = reduce(append, dictionaries, {int(s): [] for s in self.STATES})
        d_over_time = {k: np.array(v) for k, v in d_over_time.items()}
        if state_no is None:
            return d_over_time               
        return d_over_time[state_no]

    # Returns a list of the number of new vertices in the given state at each time step.
    def nr_new_over_time(self, state_no):
        states_over_time = (self.history_list == state_no)
        previous_states = np.vstack([np.zeros(states_over_time.shape[1], dtype=np.bool),
                                     states_over_time[:-1]])
        return (~previous_states & states_over_time).sum(axis=1)

    # Returns a list of the first time each vertex enters the given state, or -1 if it never does.
    def first_time(self, state_no): 
        states_over_time = (self.history_list == state_no)
        first_times = np.argmax(states_over_time, axis=0)
        first_times[(first_times == 0) & (states_over_time.sum(axis=0) == 0)] = -1   
        return first_times

    # For each state, returns a pair containing the maximum number of vertices in that state and the first time at
    # which that maximum occurs (in that order).
    def max_summary(self, states=None):
        summary = {}
        if states is None:
            states = self.STATES
        for state in states:
            nr = self.nr_over_time(state)
            summary[state] = (np.amax(nr), np.argmax(nr))
        return summary

    # Create a plot of the number of vertices in each state over time.
    def plot_history(self):
        _, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        timesteps = list(range(self.length))
        states_over_time = self.nr_over_time()
        for vertex_state in self.STATES:
            ax1.plot(timesteps, states_over_time[vertex_state],
                     label=self.STATE_LABELS[vertex_state])
        ax1.legend()

    # Make a pretty animation of the epidemic's progress.
    def make_animation(self, vertex_set, start_frame=0, end_frame=-1, frequency=1):
        x, y = vertex_set.centered_locations(vertex_set.locations[self.center_vertex]).T
        if end_frame == -1:
            end_frame = self.length
        frame_list = list(range(start_frame, end_frame, frequency))
        color_data = np.array([self.history_list[frame] for frame in frame_list])

        def update_plot(i, data, scat):
            scat.set_array(data[i])
            return scat,
        fig = plt.figure(figsize=(12, 12))
        scat = plt.scatter(x, y, c=color_data[0], s=40, vmin=0, vmax=len(self.STATES) - 1,
                           cmap=plt.get_cmap('Set2'))
        plt.axis('off')
        ani = FuncAnimation(fig,
                            update_plot,
                            frames=len(frame_list),
                            fargs=(color_data, scat),
                            interval=1000,
                            blit=True)
        self.ani = ani
        return ani

    # Save the animation to a file after making it.
    def save_animation(self, filename, fps=2):
        if not hasattr(self, "ani"):
            raise Exception("Make animation first")
        self.ani.save(filename, fps=fps)
