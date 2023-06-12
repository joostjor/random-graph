import copy

import numpy as np


class BaseSpreadingState:
    """This is a generic class which stores the current state (e.g. S/I/R) of all vertices in an infection model, and
    which handles state transitions on request.  The specific epidemic model will then inherit from this class (e.g.
    with the SEIRAppSpreadingState class).

    The state of the model is stored inside a state matrix with one row for each vertex. When a vertex is infected,
    its row of the matrix is populated by transition times to subsequent states. For example, if the model has
    infected states A, B and C which it progresses through in order, and for a given vertex the A->B column reads 4
    and the B->C column reads 6, then the vertex will spend 4 steps in state A infected and (6-4=)2 steps in state B
    before entering state C. The list of these columns is contained in the TIME_COLUMNS class variable, and the list
    of columns with other relevant information (e.g. symptomatic or asymptomatic) is contained in the STATE_COLUMNS
    class variable. The list of possible vertex states is contained in the STATES variable for logging purposes.

    State transitions are handled by the increase_time function, which just subtracts 1 from each positive entry in
    a TIME_COLUMN of the state matrix. Note that this does not include new infections --- these are handled in the
    SpreadingModel class --- only the natural progression of an infection from state to state."""
    def __init__(self,
                 number_of_vertices,
                 center_vertex=0):
        # Center_vertex is the vertex any animations of the model we produce will be centered on, typically the first
        # vertex infected.
        self.state_matrix = np.zeros((number_of_vertices, 
                                     len(self.TIME_COLUMNS) + len(self.STATE_COLUMNS),), 
                                     dtype=np.int)
        for col in self.STATE_COLUMNS:
            self.state_matrix[:, col] = self.INIT_COLUMNS[col]
        self.state_matrix = self.state_matrix
        self.center_vertex = center_vertex

    def copy(self):
        return copy.deepcopy(self)

    # Update the state by letting the specified time elapse.
    def increase_time(self, steps=1):        
        for _ in range(steps):
            for col in self.TIME_COLUMNS:
                self.state_matrix[self.state_matrix[:, col] > 0, col] -= 1

    # Returns an nparray of all vertices in the given state.
    def state_vertices(self, state_no):
        return np.where(self.state_list == state_no)[0]

    # Returns the number of vertices in the given state.
    def nr_of_state_vertices(self, state_no):
        return len(self.state_vertices(state_no))

    # state_list[i] contains the state of the i'th vertex.
    @property
    def state_list(self):
        return self.get_vertices_state()

    # Returns the total number of vertices in the graph.
    @property
    def nr_of_vertices(self):
        return len(self.state_matrix)

    # Returns the number of vertices in the given state (from STATES). Note this is overridden in SEIRAppSpreadingState.
    def get_vertices_state(self):        
        states = np.empty(self.nr_of_vertices, dtype=np.int8)  
        for state in self.STATES:
            states[self.state_mask(state)] = state
        return states

    @property
    def state_matrix(self):
        return self._state_matrix

    @state_matrix.setter
    def state_matrix(self, value):
        self._state_matrix = value

    # Returns an array of all vertices capable of infecting others, e.g. I in SITS. Used in the infection step.
    @property
    def infectious_vertices(self):
        raise NotImplementedError("infectious_vertices not implemented")

    # Returns an array of all vertices capable of being infected, e.g. S in SIR. Used in the infection step.
    @property 
    def susceptible_vertices(self):
        raise NotImplementedError("susceptible_vertices not implemented")

    # Returns true if the model's state is still evolving (i.e. we want to keep simulating it).
    @property
    def state_active(self):
        raise NotImplementedError("state_active not implemented")

    # Returns the list of all column indices of the state matrix that need to be decremented each time step (e.g. for
    # I -> R transitions).
    @property
    def TIME_COLUMNS(self):
        raise NotImplementedError("TIME_COLUMNS not implemented")

    # Returns the list of all column indices of the state matrix that contain constant boolean information (e.g.
    # whether the vertex has previously been infected in SIR).
    @property
    def STATE_COLUMNS(self):
        raise NotImplementedError("STATE_COLUMNS not implemented")

    # Returns a dictionary of the desired initial values for each state column (when no vertices are infected).
    # Note that numpy will cast Boolean values to integers in the state matrix, so best to avoid using them.
    @property
    def INIT_COLUMNS(self):
        raise NotImplementedError("INIT_COLUMNS not implemented")

    # Returns an IntEnum for the possible states (e.g. S/I/R) of each vertex.
    @property
    def STATES(self):
        raise NotImplementedError("State list not implemented.")

    # Returns a dictionary containing the name of each state in STATES, for logging purposes.
    @property
    def STATE_LABELS(self):
        raise NotImplementedError("State labels not implemented.")

    # Returns a vector x such that x[i] = 1 if vertex i is in state state_no, and x[i] = 0 otherwise. Used to map
    # between the state matrix and actual vertex states. Note this is not actually used in the SEIRAppSpreadingState
    # class (we instead redefine get_vertices_state).
    def state_mask(self, state_no):
        raise NotImplementedError("")

    # Infects the vertices contained in the list indices with future infection trajectories contained in the list
    # infections. (This history should be randomly generated in the corresponding SpreadingModel class.)
    def infect_vertices(self, indices, infections):
        raise NotImplementedError("infect_vertices is not implemented")
