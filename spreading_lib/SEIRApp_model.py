from enum import IntEnum
import random
import re

import numpy as np

from .base_model import SpreadingModel
from .base_state import BaseSpreadingState
from scipy.sparse import csr_matrix

# This file contains all the inherited classes for the main model of the paper.


# This class contains enums for the TIME and STATE columns of the state matrix for SEIRAppSpreadingState. On
# initialisation, FLAG_WILL_SYMPTOM and FLAG_HAS_APP are set randomly, and all other columns are false or zero.
# On infection, a vertex v enters state E, FLAG_NON_SUSCEPTIBLE is set, and all the TIME columns are filled out
# via a SEIRAppInfection; this will be generated randomly by SEIRAppModel.infection_generator. After TIME_UNTIL_INFECTED
# time steps, v enters state I. If FLAG_WILL_SYMPTOM is true, then after TIME_UNTIL_SYMPTOMS steps v will enter state Se
# and self-quarantine, FLAG_WILL_SYMPTOM will be unset, and FLAG_HAS_SYMPTOMED will be set. Whether this happens or not,
# v enters state R after TIME_UNTIL_REMOVED steps. If v self-quarantines at any point, TIME_UNTIL_DEQUARANTINED will be
# set, and v will remain in quarantine for TIME_UNTIL_DEQUARANTINED steps. If a neighbour of v self-quarantines due
# to developing symptoms, then v will self-quarantine if FLAG_HAS_APP is set. Infection can only occur from
# non-quarantined vertices in state I to non-quarantined vertices in state S.
class SEIRAppVertexStateColumn(IntEnum):
    TIME_UNTIL_INFECTED      = 0
    TIME_UNTIL_REMOVED       = 1
    TIME_UNTIL_SYMPTOMS      = 2
    TIME_UNTIL_DEQUARANTINED = 3
    FLAG_NON_SUSCEPTIBLE     = 4
    FLAG_HAS_APP             = 5
    FLAG_WILL_SYMPTOM        = 6
    FLAG_HAS_SYMPTOMED       = 7


# This class maps the states of the SEIRAppSpreadingState model to numerical values. It's important that every
# quarantined state is one more than its unquarantined counterpart except symptomatic (this lets us optimise the
# state extraction).
class SEIRAppStates(IntEnum):
    SUSCEPTIBLE   = 0
    Q_SUSCEPTIBLE = 1
    EXPOSED       = 2
    Q_EXPOSED     = 3
    INFECTED      = 4
    Q_INFECTED    = 5
    REMOVED       = 6
    Q_REMOVED     = 7
    SYMPTOMATIC   = 8


# This class contains the state data for each infection of a given round, given as a lists of variables whose values
# will be randomised. Note both times here should be measured from the START of the infection.
class SEIRAppInfection:
    def __init__(self,
                 time_until_infected,
                 time_until_removed,
                 time_until_symptoms,  # Will be ignored for vertices which don't develop severe symptoms
                 flag_will_symptom):
        self.time_until_infected = time_until_infected
        self.time_until_removed = time_until_removed
        self.time_until_symptoms = time_until_symptoms
        self.flag_will_symptom = flag_will_symptom


# This is the inherited class of SpreadingState specific to our paper, holding the current state and handling
# "internal" state transitions of the model. Important note: In this code, we call the "Se" state in the paper (for
# "severely symptomatic") by a different name: "Sy" (for "symptomatic"). There is no difference between the two states.
class SEIRAppSpreadingState(BaseSpreadingState):
    @property
    def TIME_COLUMNS(self):
        return [
            SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED,
            SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED,
            SEIRAppVertexStateColumn.TIME_UNTIL_SYMPTOMS,
            SEIRAppVertexStateColumn.TIME_UNTIL_DEQUARANTINED
        ]

    @property
    def STATE_COLUMNS(self):
        return [
            SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE,
            SEIRAppVertexStateColumn.FLAG_HAS_APP,
            SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM,
            SEIRAppVertexStateColumn.FLAG_HAS_SYMPTOMED
        ]

    @property
    def INIT_COLUMNS(self):
        return {SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE:0,
                SEIRAppVertexStateColumn.FLAG_HAS_APP:0,  # This will be re-initialised later in __init__
                SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM:0,
                SEIRAppVertexStateColumn.FLAG_HAS_SYMPTOMED:0}

    @property
    def STATES(self):
        return SEIRAppStates

    @property
    def STATE_LABELS(self):
        return {
            self.STATES.SUSCEPTIBLE: "Susceptible",
            self.STATES.Q_SUSCEPTIBLE: "Susceptible and quarantined",
            self.STATES.EXPOSED: "Exposed",
            self.STATES.Q_EXPOSED: "Exposed and quarantined",
            self.STATES.INFECTED: "Infected",
            self.STATES.Q_INFECTED: "Infected and quarantined",
            self.STATES.REMOVED: "Removed",
            self.STATES.Q_REMOVED: "Removed and quarantined",
            self.STATES.SYMPTOMATIC: "Severely symptomatic"
        }

    def __init__(self,
                 number_of_vertices,
                 app_rate,
                 quarantine_len,
                 method="random",  # can be changed to "recommender", "degree", "recommend_friends_with_p=0..."
                 graph=None,
                 center_vertex=0):
        super().__init__(number_of_vertices, center_vertex)
        recommender_regex = re.compile(r"recommend_friends_with_p=(?P<rec_rate>0.\d+)")
        if method == "random":
            app_users = np.where(np.random.random(number_of_vertices) < app_rate)[0]                            
        elif graph is None:
            raise Exception(f"Specify graph for method {method}")
        
        elif method == "degree":
            degrees = graph.degrees
            nr_app_users = int(app_rate*len(degrees))
            app_users = np.argpartition(-degrees, nr_app_users)[:nr_app_users]
        
        elif method == "recommender":
            app_users_native = np.where(np.random.random(graph.nr_of_nodes) < app_rate)[0]
            neighbors_per_user = dict()
            for user, neighbor in zip(*graph.distance_matrix[app_users_native].nonzero()):
                if user not in neighbors_per_user:
                    neighbors_per_user[user] = [neighbor]
                else:
                    neighbors_per_user[user] += [neighbor]
            app_users_recommended = np.array([random.choice(neighbors) 
                                              for neighbors in neighbors_per_user.values()])
            app_users = np.unique(np.concatenate([app_users_recommended, 
                                                  app_users_native]))
        elif recommender_regex.match(method) is not None:
            recommender_rate = float(recommender_regex.match(method).groupdict()["rec_rate"])
            app_users_native = np.where(np.random.random(graph.nr_of_nodes) < app_rate)[0]
            _, neighbors = graph.distance_matrix[app_users_native].nonzero()
            app_users_recommended = neighbors[np.random.random(size=neighbors.size) < recommender_rate]
            app_users = np.unique(np.concatenate([app_users_recommended,
                                                  app_users_native]))
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")
        self.state_matrix[app_users,SEIRAppVertexStateColumn.FLAG_HAS_APP] = 1
        self.quarantine_len = quarantine_len

    @property
    def unquarantined_mask(self):
        return self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_DEQUARANTINED] == 0

    @property
    def unquarantined_vertices(self):
        return np.where(self.unquarantined_mask)[0]

    # Vertices which should be quarantined in the current time step due to symptoms developing
    @property
    def vertices_to_self_quarantine(self):
        return np.where((self.state_matrix[:,SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM]) &
                        (self.state_matrix[:,SEIRAppVertexStateColumn.TIME_UNTIL_SYMPTOMS] == 0))[0]

    # Vertices which have just developed symptoms and which should notify neighbours via the app.
    @property 
    def vertices_to_notify_others(self):
        return np.where((self.state_matrix[:,SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM]) &
                        (self.state_matrix[:,SEIRAppVertexStateColumn.TIME_UNTIL_SYMPTOMS] == 0) &
                        (self.state_matrix[:,SEIRAppVertexStateColumn.FLAG_HAS_APP]))[0]
    
    # Vertices which should be asked to quarantine *if* they're adjacent to a vertex which has just self-quarantined
    @property
    def vertices_to_app_quarantine(self):
        return np.where((self.state_matrix[:,SEIRAppVertexStateColumn.FLAG_HAS_APP]) &
                        (self.state_matrix[:,SEIRAppVertexStateColumn.TIME_UNTIL_DEQUARANTINED] == 0))[0]

    @property
    def susceptible_vertices(self):
        return np.where((self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE] == 0) &
                        self.unquarantined_mask)[0]

    @property
    def infectious_vertices(self):
        return np.where((self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED] > 0) &
                        (self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED] == 0) &
                        self.unquarantined_mask)[0]

    @property
    def S_vertices(self):
        return np.where(self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE] == 0)[0]

    @property
    def E_vertices(self):
        return np.where(self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED] > 0)[0]

    @property
    def I_vertices(self):
        return np.where((self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED] > 0) &
                        (self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED] == 0) &
                        ~(self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_HAS_SYMPTOMED]))[0]

    # Again, recall that state Sy in the code corresponds to state Se in the paper.
    @property
    def Sy_vertices(self):
        return np.where((self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED] > 0) &
                        (self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_HAS_SYMPTOMED]))[0]

    @property
    def R_vertices(self):
        return np.where((self.state_matrix[:, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED] == 0) &
                        (self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE] == 1))[0]

    @property
    def state_active(self):
        return (self.state_matrix[:, [SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED,
                                      SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED]] > 0).any()

    # Proportion of vertices which have the app.
    @property 
    def app_coverage(self):
        return self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_HAS_APP].mean()

    # Number of vertices which have the app.
    @property
    def app_users(self):
        return np.where(self.state_matrix[:, SEIRAppVertexStateColumn.FLAG_HAS_APP] == 1)[0]

    def get_vertices_state(self, vertices=None):
        if vertices is None:
            vertices = np.arange(self.nr_of_vertices, dtype=np.int)
        states = np.empty(vertices.shape, dtype=np.int8)
        states[self.S_vertices] = self.STATES.Q_SUSCEPTIBLE
        states[self.E_vertices] = self.STATES.Q_EXPOSED
        states[self.I_vertices] = self.STATES.Q_INFECTED
        states[self.R_vertices] = self.STATES.Q_REMOVED
        states[self.Sy_vertices] = self.STATES.SYMPTOMATIC
        # If the quarantined state is x, then the unquarantined state is always x-1.
        # Symptomatic vertices are always quarantined, so they're unaffected.
        states -= self.unquarantined_mask
        return states

    def infect_vertices(self,
                        indices,
                        infections,
                        debug=False):
        if len(indices) == 0:
            return
        self.state_matrix[indices, SEIRAppVertexStateColumn.TIME_UNTIL_INFECTED] = infections.time_until_infected
        self.state_matrix[indices, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED] = infections.time_until_removed
        self.state_matrix[indices, SEIRAppVertexStateColumn.TIME_UNTIL_SYMPTOMS] = infections.time_until_symptoms
        self.state_matrix[indices, SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM] = infections.flag_will_symptom
        self.state_matrix[indices, SEIRAppVertexStateColumn.FLAG_NON_SUSCEPTIBLE] = 1

    def quarantine_vertices(self, self_indices, app_indices):
        if len(app_indices) > 0:
            self.state_matrix[app_indices, SEIRAppVertexStateColumn.TIME_UNTIL_DEQUARANTINED] = self.quarantine_len
        if len(self_indices) > 0:
            self.state_matrix[self_indices, SEIRAppVertexStateColumn.TIME_UNTIL_DEQUARANTINED] = \
                self.state_matrix[self_indices, SEIRAppVertexStateColumn.TIME_UNTIL_REMOVED]
            self.state_matrix[self_indices, SEIRAppVertexStateColumn.FLAG_WILL_SYMPTOM] = False
            self.state_matrix[self_indices, SEIRAppVertexStateColumn.FLAG_HAS_SYMPTOMED] = True


# This is the inherited class of SpreadingModel specific to our paper, handling infection dynamics and setup specific
# to the model in our paper.
class SEIRAppModel(SpreadingModel):
    def __init__(self,
                 infection_prob,      # Probability an I vertex infects an S neighbour each time step
                 infection_rate,      # Probability an E vertex enters state I each time step
                 removal_rate,        # Probability an I vertex enters state R each time step
                 app_rate,            # Probability a given vertex has the app                 
                 symptom_prob,        # Probability an I vertex develops symptoms and goes into quarantine spontaneously
                 symptom_rate,        # Probability a symptomatic I vertex goes into quarantine each time step
                 quarantine_len=14,   # Duration of quarantine in time steps
                 method="random",     # Way to select app users: random, degree (highest degree nodes), recommender
                 graph=None):         # If method is not random, graph needs to be specified to determine users
        super().__init__()
        self.infection_prob = infection_prob
        self.infection_rate = infection_rate
        self.removal_rate = removal_rate
        self.app_rate = app_rate
        self.method = method
        self.symptom_prob = symptom_prob
        self.symptom_rate = symptom_rate
        if symptom_rate == 0:                      # In this case symptomatic_removal_rate is never used.
            self.symptomatic_removal_rate = 1
        elif 1/removal_rate - 1/symptom_rate < 1:  # This shouldn't ever happen under our planned parameter choices.
            raise Exception("Removal rate is higher than symptom rate; correct behaviour is undefined.")
        else:
            self.symptomatic_removal_rate = 1/(1/removal_rate - 1/symptom_rate)
        self.quarantine_len = quarantine_len
        self.app_coverages = list()
        self.graph = graph
        self.parameters = ["infection_prob",                           
                           "infection_rate",
                           "removal_rate",
                           "app_rate",
                           "method",
                           "app_coverages",
                           "symptom_prob",
                           "symptom_rate",
                           "quarantine_len"]

    def determine_infections(self, distances):
        return np.random.random(len(distances)) < self.infection_prob

    def infection_generator(self, number_of_infections):
        time_until_infected = np.random.geometric(self.infection_rate, number_of_infections)
        flag_will_symptom = np.random.random(number_of_infections) < self.symptom_prob
        indices_will_symptom = np.where(flag_will_symptom)[0]
        time_until_symptoms = time_until_infected + np.random.geometric(self.symptom_rate, number_of_infections)
        time_until_removed = time_until_infected + np.random.geometric(self.removal_rate, number_of_infections)
        time_until_removed[indices_will_symptom] = time_until_symptoms[indices_will_symptom] + np.random.geometric(
            self.symptomatic_removal_rate, len(indices_will_symptom))

        return SEIRAppInfection(time_until_infected, time_until_removed, time_until_symptoms, flag_will_symptom)

    @property
    def spreading_state_class(self):
        return SEIRAppSpreadingState

    # Only change from default_infection_state_maker is passing quarantine_len and app_rate to the state class.
    def infection_state_maker(self, 
                              number_of_vertices, 
                              number_of_infections=1):
        def maker():            
            initial_state = self.spreading_state_class(number_of_vertices=number_of_vertices, 
                                                       app_rate=self.app_rate, 
                                                       quarantine_len=self.quarantine_len,
                                                       method=self.method,
                                                       graph=self.graph)
            self.app_coverages.append(initial_state.app_coverage) 
            
            infection = self.infection_generator(number_of_infections)
            vertex_indices = np.random.choice(np.arange(number_of_vertices),
                                              size=number_of_infections,
                                              replace=False)           
            initial_state.infect_vertices(vertex_indices, infection)
            initial_state.center_vertex = vertex_indices[0]
            return initial_state
        return maker

    def _single_round(self,
                      spreading_state,
                      distance_matrix):
        super()._single_round(spreading_state, distance_matrix)

        self_quarantine_vertices = spreading_state.vertices_to_self_quarantine
        vertices_to_notify_others = spreading_state.vertices_to_notify_others
        # app_quarantine_vertices is a list of all app_vertices adjacent to at least one quarantine_vertex.
        if not hasattr(self, "app_matrix"):
            self.app_users = spreading_state.app_users
            self.app_users_rev = np.zeros(distance_matrix.shape[0], dtype=np.int)
            for app_index, distance_index in enumerate(self.app_users):
                self.app_users_rev[distance_index] = app_index           
            self.app_users_rev = csr_matrix([self.app_users_rev])
            self.app_matrix = distance_matrix[self.app_users].T[self.app_users].T        
        vertices_to_notify_others_app = self.app_users_rev[:, vertices_to_notify_others].toarray().flatten()       
        _, nonzero_indices_cols = self.app_matrix[vertices_to_notify_others_app].nonzero()        
        app_quarantine_vertices = np.unique(self.app_users[nonzero_indices_cols])
        spreading_state.quarantine_vertices(self_quarantine_vertices, app_quarantine_vertices)

    def single_run(self,
                   make_initial_state,
                   graph_determiner,
                   max_rounds,
                   min_infected=0):
        history = super().single_run(make_initial_state,
                                     graph_determiner,
                                     max_rounds,
                                     min_infected)
        delattr(self, "app_matrix")
        return history
