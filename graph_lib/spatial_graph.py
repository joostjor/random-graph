import copy
from functools import reduce
import itertools
import json
from pathlib import Path
import tarfile
import tempfile
import time

import networkx as nx
from numba import jit, prange
import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, triu, find, load_npz, save_npz


# This class contains general functions for dealing with a vertex set laid out in a d-dimensional torus (as found in
# GIRGs), e.g. generating a vertex set from a Poisson point process or determining the distance between two points.
class VertexSet:
    def __init__(self,
                 locations,
                 weights,
                 width=None,
                 time_generated=None,
                 centre_vertex_index=None):
        self.locations = locations.astype(np.float)
        self.weights = weights
        if time_generated is None:
            time_generated = int(time.time())
        self.time_generated = time_generated
        self._width = width
        self._centre_vertex_index = centre_vertex_index

    @staticmethod
    def _generate_locations(size,
                            dimension,
                            torus_width,
                            add_origin=False):
        """Returns (size*dimension) numpy array with points on the d-dimensional torus
        of length size^(1/dimension)"""
        if add_origin:
            return np.vstack([np.ones((1, dimension)) * torus_width/2,
                              np.random.random(size=(size-1, dimension)) * torus_width])
        else:
            return np.random.random(size=(size, dimension)) * torus_width

    @staticmethod
    def _torus_width(vertex_size, dimension):
        return vertex_size**(1/dimension)

    @classmethod
    def from_poisson_point_process(cls,
                                   number_of_vertices,
                                   dimension,
                                   add_origin=False,
                                   width=None):
        if width is None:
            width = cls._torus_width(number_of_vertices,
                                     dimension)
        return cls(cls._generate_locations(number_of_vertices,
                                           dimension,
                                           width,
                                           add_origin=add_origin),
                   np.random.random(size=number_of_vertices), 
                   width=width,
                   centre_vertex_index=0)

    @classmethod
    def on_grid(cls,
                width,
                dimension, 
                centre_first=False):
        if centre_first:
            array = [width // 2] + list(range(width//2)) + list(range(width//2 + 1, width))
        else:
            array = list(range(width))
        locations = np.array(list(itertools.product(array, repeat=dimension)))
        return cls(locations,
                   np.random.random(size=width**dimension))

    @classmethod
    def from_file(cls, filename):
        return cls(**np.load(filename))

    @property
    def number_of_vertices(self):
        return self.locations.shape[0]

    @property
    def average_weight(self):
        return self.weights.mean()

    @property
    def dimension(self):
        return self.locations.shape[1]

    @property
    def torus_width(self):  
        return self._width or self._torus_width(self.number_of_vertices, self.dimension)

    @property
    def centre_vertex_index(self):
        if self._centre_vertex_index is None:
            try:
                return [i for i, l in enumerate(self.locations.tolist()) 
                        if l==[self.torus_width//2]*self.dimension][0]            
            except:
                raise ValueError("Centre vertex not existing")
        else:
            return self._centre_vertex_index
    
    def describe(self):
        return {
            "number_of_vertices": self.number_of_vertices,
            "average_weight": self.average_weight,
            "dimension": self.dimension,
            "torus_width": self.torus_width
        }

    def centered_locations(self, centre):
        torus_width = self.torus_width
        diff = self.locations - centre
        overshoot_mask = diff > torus_width/2
        diff[overshoot_mask] -= torus_width
        undershoot_mask = diff < -torus_width/2
        diff[undershoot_mask] += torus_width
        return diff

    def save(self, filename):
        filename = str(filename)
        if not filename.endswith(".npz"):
            filename += ".npz"
        np.savez(filename,
                 locations=self.locations,
                 weights=self.weights,
                 time_generated=self.time_generated)


@jit(nopython=True)
def distance_from(other_locations,
                  vertex_location,
                  torus_width,
                  on_torus,
                  p=2):
    abs_diff = np.abs(vertex_location - other_locations)
    if on_torus:
        torus_diff = np.minimum(torus_width - abs_diff,
                                abs_diff)
        return np.sum(torus_diff**p, axis=1)**(1/p)
    else:
        return np.sum(abs_diff**p, axis=1)**(1/p)


def _pairwise_minimum_matrices(matrix1, matrix2):
    matrix = matrix1 + matrix2
    indices_overlap = (matrix1 > 0).multiply((matrix2 > 0)).nonzero()
    matrix[indices_overlap] = np.minimum(matrix1[indices_overlap].A, 
                                         matrix2[indices_overlap].A)
    return matrix


# This is a generic class for generating and manipulating a random graph (possibly with a spatial component) from a
# given VertexSet, with the option to export to networkx format for more in-depth work. Our classes for both the GIRG
# model and the configuration model will inherit from this class, only implementing methods for sampling. Note that much
# of the functionality of the class is not used in this paper.
class SpatialGraph:
    def __init__(self,
                 vertex_set,
                 distance_matrix=None,
                 on_torus=True,
                 timestamps=None):
        self.vertex_set = vertex_set
        self.on_torus = on_torus
        self._distance_matrix = distance_matrix
        if timestamps is None:
            timestamps = [{"event_type":"generated", "time": int(time.time())}]
        self.timestamps = timestamps
        self._parameters_base = ["on_torus", "timestamps"]
        self._parameters = []

    def union(self, *other_graphs):
        """
        create graph on vertex set of the original graph,
        where an edge exists if it is present in one of the other graphs,
        the distance between two is the minimal distance in all graphs
        """
        for g in other_graphs:
            assert self.nr_of_nodes == g.nr_of_nodes
        distance_matrix = reduce(lambda a, b: _pairwise_minimum_matrices(a, b), 
                                 (g.distance_matrix for g in other_graphs), 
                                 csr_matrix((self.nr_of_nodes,)*2))
        timestamps = [{"event_type": "union", 
                       "time": int(time.time()), 
                       "graphs": [g.describe() for g in other_graphs]}]
        return SpatialGraph(self.vertex_set, 
                            distance_matrix,
                            timestamps=timestamps)
    
    @classmethod 
    def make_line_graph(cls, nr_of_vertices):
        vertex_loc = np.vstack([np.arange(nr_of_vertices), np.zeros(nr_of_vertices)]).T
        vertex_weights = np.zeros(nr_of_vertices)
        vertex_set = VertexSet(vertex_loc, vertex_weights)
        distance_matrix = csr_matrix(np.diag(np.ones(nr_of_vertices-1), k=1) + np.diag(np.ones(nr_of_vertices-1), k=-1))
        return cls(vertex_set=vertex_set,
                   distance_matrix=distance_matrix)
    
    @classmethod
    def make_grid_graph(cls, dimension, width, on_torus=False):
        graph = nx.generators.grid_graph(dim=[width for _ in range(dimension)], periodic=on_torus)
        vertex_locations = np.array(list(graph.nodes()))
        vertex_weights = np.full(vertex_locations.shape[0], np.nan)
        distance_matrix = nx.to_scipy_sparse_matrix(graph)
        vertex_set = VertexSet(vertex_locations, vertex_weights)
        return cls(vertex_set, distance_matrix)

    @classmethod
    def make_2d_grid_infinity_norm(cls, width, on_torus=False):
        def coord_to_integer(row, col):
            return row*width + col

        def integer_to_coord(integer):
            return [integer // width, integer % width]

        vectors = np.array([[0,1], [1,1], [1,0], [-1,1]])

        def determine_neighbors(integer):
            potential_neighbors = np.asarray(integer_to_coord(integer)) + vectors
            if on_torus:
                neighbor_locations = (potential_neighbors % width).tolist()
            else:
                mask = ((potential_neighbors >= 0)
                        & (potential_neighbors < width)).all(axis=1)
                neighbor_locations = (potential_neighbors[mask]).tolist()
            neighbors = [coord_to_integer(*neighbor) for neighbor in neighbor_locations]
            return neighbors

        edge_list = [[i, n] for i in range(width**2) for n in determine_neighbors(i)]
        distance_matrix = csr_matrix((np.ones(len(edge_list)), np.asarray(edge_list).T),
                                     shape=(width**2, width**2))
        distance_matrix += distance_matrix.T
        vertex_locations = np.array([integer_to_coord(i) for i in range(width**2)])
        vertex_weights = np.ones(width**2)
        return cls(VertexSet(vertex_locations, vertex_weights), distance_matrix, on_torus)

    def to_networkx_graph(self, centre_vertex=None, accept_empty=False):
        if centre_vertex is None:
            locations = self.vertex_set.locations
        else:
            locations = self.vertex_set.centered_locations(self.vertex_set.locations[centre_vertex])
        if accept_empty:
            try:
                graph_nx = nx.from_scipy_sparse_matrix(self.distance_matrix)
            except ValueError: 
                graph_nx = nx.from_scipy_sparse_matrix(csr_matrix((len(self.vertex_set.locations),
                                                                   len(self.vertex_set.locations))))
        else:
            graph_nx = nx.from_scipy_sparse_matrix(self.distance_matrix)                                                       
        nx.set_node_attributes(graph_nx, dict(enumerate(locations.tolist())), "location")
        return graph_nx

    @property
    def parameters(self):
        return self._parameters + self._parameters_base

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = self._parameters_base + parameters

    @property
    def parameter_dict(self):
        return dict([(param, getattr(self, param))
                     for param in self.parameters])
    
    @property
    def centre_vertex_index(self):
        return self.vertex_set.centre_vertex_index
                         

    def sample_graph(self):
        raise NotImplementedError("")

    def save_graph(self, filename):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "distance_matrix.npz", self.distance_matrix)
            self.vertex_set.save(tmp_path  / "vertex_set.npz")
            with open(tmp_path / "params.json", "w") as f:
                param_dict = copy.deepcopy(self.parameter_dict)
                param_dict["class"] = type(self).__name__
                json.dump(param_dict, f, indent=4, sort_keys=True)
            with tarfile.open(filename, "w:gz") as tar:
                tar.add(tmp_path / "distance_matrix.npz", arcname="distance_matrix.npz")
                tar.add(tmp_path / "vertex_set.npz", arcname="vertex_set.npz")
                tar.add(tmp_path / "params.json", arcname="params.json")

    def draw(self, node_size=10, pos=None, highlight_component_vertex=None, node_color="r"):
        graph_nx = self.to_networkx_graph(centre_vertex=highlight_component_vertex)
        if pos is None:
            pos = dict(enumerate(self.vertex_set.locations.tolist()))
        _node_color = 'b' if highlight_component_vertex is not None else node_color
        nx.draw_networkx_nodes(graph_nx,
                               pos=pos,
                               node_size=node_size if highlight_component_vertex is None else 5,
                               node_color=_node_color)
        nx.draw_networkx_edges(graph_nx, pos=pos, edge_color='b')
        if highlight_component_vertex is not None:
            connected_component = [cc for cc in nx.components.connected_components(graph_nx) 
                                   if highlight_component_vertex in cc][0]
            subgraph = graph_nx.subgraph(connected_component)
            nx.draw_networkx_nodes(subgraph, 
                                   pos=pos, 
                                   nodelist=[i for i in connected_component if i!=highlight_component_vertex],
                                   node_size=5, 
                                   node_color=_node_color)
            nx.draw_networkx_nodes(subgraph, 
                                   pos=pos, 
                                   nodelist=[highlight_component_vertex],
                                   node_size=40, 
                                   node_color='g')
            nx.draw_networkx_edges(subgraph, pos=pos, edge_color='r')


    def draw_1d(self, highlight_component_vertex=None, tau=10, accept_empty=False):        
        graph_nx = self.to_networkx_graph(centre_vertex=highlight_component_vertex, accept_empty=accept_empty)
        pos = dict(enumerate(np.vstack([self.vertex_set.locations.flatten(), 
                                        5*np.log(self.vertex_set.weights**(-1/(tau-1)))]).T.tolist()))
        nx.draw_networkx_nodes(graph_nx,
                               pos=pos,
                               node_size=40 if highlight_component_vertex is None else 5,
                               node_color='r' if highlight_component_vertex is None else 'b')
        nx.draw_networkx_edges(graph_nx, pos=pos, edge_color='b')

        if highlight_component_vertex is not None:
            connected_component = [cc for cc in nx.components.connected_components(graph_nx) 
                                   if highlight_component_vertex in cc][0]
            subgraph = graph_nx.subgraph(connected_component)
            nx.draw_networkx_nodes(subgraph, 
                                   pos=pos, 
                                   nodelist=[i for i in connected_component if i!=highlight_component_vertex],
                                   node_size=5, 
                                   node_color='r')
            nx.draw_networkx_nodes(subgraph, 
                                   pos=pos, 
                                   nodelist=[highlight_component_vertex],
                                   node_size=40, 
                                   node_color='g')
            nx.draw_networkx_edges(subgraph, pos=pos, edge_color='r')    
    
    def draw_in_circle(self, node_size=10):
        x = np.cos(np.arange(0, self.nr_of_nodes)*2*np.pi / self.nr_of_nodes)
        y = np.sin(np.arange(0, self.nr_of_nodes)*2*np.pi / self.nr_of_nodes)
        pos = dict(enumerate(np.vstack([x, y]).T.tolist()))
        self.draw(node_size, pos)

    def draw_spring_layout(self, node_size=10, iterations=50):
        graph_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)
        pos = nx.drawing.layout.spring_layout(graph_nx, iterations=iterations)
        self.draw(node_size, pos)

    @classmethod
    def load_graph(cls, filename):
        with tarfile.open(filename, "r:gz") as tar:
            vertex_set = VertexSet.from_file(tar.extractfile("vertex_set.npz"))
            distance_matrix = load_npz(tar.extractfile("distance_matrix.npz"))
            parameters = json.load(tar.extractfile("params.json"))
        assert parameters["class"] == cls.__name__
        parameters.pop("class", None)
        return cls(vertex_set=vertex_set,
                   distance_matrix=distance_matrix,
                   **parameters)

    # TODO: Move interventions outside class to other file
    def edge_percolated_graph(self, retention_probability):
        sources, targets, values = find(triu(self.distance_matrix))
        indices_to_keep = np.random.random(size=len(values)) < retention_probability
        matrix = csr_matrix((values[indices_to_keep],
                             (sources[indices_to_keep],targets[indices_to_keep])),
                            shape=self.distance_matrix.shape)
        parameter_dict = copy.deepcopy(self.parameter_dict)
        if not "timestamps" in parameter_dict:
            parameter_dict["timestamps"] = list()
        parameter_dict["timestamps"].append({"event_type":{"edge_percolation":retention_probability},
                                             "time": int(time.time())})
        return self.__class__(vertex_set=self.vertex_set,
                              distance_matrix=matrix.T + matrix,
                              **parameter_dict)

    def truncate_long_edges(self, exponential_mean):
        distance_matrix_upper = triu(self.distance_matrix)
        row_indices, col_indices, data = find(distance_matrix_upper)
        remaining_edge_flags = data < np.random.exponential(exponential_mean, size=data.shape)
        distance_matrix_new = csr_matrix((data[remaining_edge_flags], (row_indices[remaining_edge_flags],
                                                                       col_indices[remaining_edge_flags])),
                                         shape=self.distance_matrix.shape)
        parameter_dict = copy.deepcopy(self.parameter_dict)
        parameter_dict["timestamps"].append({"event_type":{"long_edge_truncation": exponential_mean},
                                             "time": int(time.time())})
        return self.__class__(vertex_set=self.vertex_set,
                              distance_matrix=distance_matrix_new + distance_matrix_new.T,
                              **parameter_dict)

    def truncate_hubs(self, degree_threshold):
        hub_indices = (self.degrees > degree_threshold).nonzero()[0]

        def truncate_single_hub(new_matrix, hub_index):
            degree = (new_matrix[hub_index] > 0).sum()
            if degree <= degree_threshold:
                return new_matrix
            row_indices_shifted, col_indices, data = find(new_matrix[hub_index])
            row_indices = np.full(shape=row_indices_shifted.shape,
                                  fill_value=hub_index)
            indices_keep = np.random.choice(np.arange(degree),
                                            size=degree_threshold,
                                            replace=False)
            zeros = np.zeros(data.shape[0], dtype=np.bool)
            zeros[indices_keep] = True
            new_matrix[np.concatenate([row_indices[~zeros],
                                       col_indices[~zeros]]),
                       np.concatenate([col_indices[~zeros],
                                       row_indices[~zeros]])] = 0
            return new_matrix
        distance_matrix_new = csr_matrix(reduce(truncate_single_hub,
                                                hub_indices,
                                                lil_matrix(copy.deepcopy(self.distance_matrix))))
        parameter_dict = copy.deepcopy(self.parameter_dict)
        parameter_dict["timestamps"].append({"event_type":{"hub_truncation": degree_threshold},
                                             "time": int(time.time())})
        return self.__class__(vertex_set=self.vertex_set,
                              distance_matrix=distance_matrix_new + distance_matrix_new.T,
                              **parameter_dict)

    @property
    def distance_matrix(self):
        if self._distance_matrix is None:
            raise ValueError("Graph not sampled, so distance matrix does not exist")
        return self._distance_matrix

    @property
    def adjacency_matrix(self):
        return self.distance_matrix.astype(np.bool)

    @property
    def nr_of_edges(self):
        return self.adjacency_matrix.sum() / 2

    @property
    def average_degree(self):
        return self.degrees.mean()

    @property
    def average_degree_star(self):
        return -1 + (self.degrees**2).sum() / (2*self.nr_of_edges)

    @property
    def nr_of_nodes(self):
        return self.distance_matrix.shape[0]

    @property
    def degrees(self):
        return np.asarray(self.adjacency_matrix.sum(axis=1)).flatten()

    @property 
    def triangles_per_vertex(self):
        adj_matrix = self.adjacency_matrix.astype(np.int)
        return (adj_matrix@adj_matrix@adj_matrix).diagonal()
    
    @property
    def clustering_per_vertex(self):
        return self.triangles_per_vertex / (self.degrees*(self.degrees-1))
    
    
    def describe(self):
        summary_dict = self.vertex_set.describe()
        summary_dict["average_degree"] = self.average_degree
        summary_dict["on_torus"] = self.on_torus
        summary_dict["class"] = type(self).__name__
        for k, v in self.parameter_dict.items():
            summary_dict[k] = v
        return summary_dict


import networkx as nx
from numba import jit, prange
import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, triu, find, load_npz, save_npz



# We use this class for geometric inhomogeneous random graphs.
class InterpolatingKSRG(SpatialGraph):
    def __init__(self,
                 vertex_set,
                 alpha,
                 tau,
                 sigma,
                 c1=1,
                 c2=1,
                 **kwargs):
        super().__init__(vertex_set,
                         **kwargs)
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2
        self.parameters = ["alpha", "tau", "sigma", "c1", "c2"]

    # Helper function that samples a list of edges for a new GIRG with the appropriate parameters.
    @staticmethod
    @jit(nopython=True)
    def _fast_edge_list(vertex_locations,
                        vertex_weights,
                        torus_width,
                        c1,c2,gamma,alpha,sigma,
                        on_torus,
                        edge_rvs=None):
        nr_of_vertices, dimension = vertex_locations.shape
        edge_list = []
        for v1 in prange(nr_of_vertices-1):
            other_locations = vertex_locations[v1+1:]
            other_weights = vertex_weights[v1+1:]
            distances = distance_from(other_locations,
                                      vertex_locations[v1],
                                      torus_width,
                                      on_torus)
            prob = c1*np.minimum(1, c2*(((np.minimum(vertex_weights[v1]**(-gamma), other_weights**(-gamma))**sigma)*
                                        np.maximum(vertex_weights[v1]**(-gamma), other_weights**(-gamma)))
                                 / (distances**dimension))**alpha)
            if edge_rvs is None:
                edge_rvs_v1 = np.random.random(len(distances))
            else:
                edge_rvs_v1 = edge_rvs[v1, v1+1:]
            target_nodes_indices = (edge_rvs_v1 < prob).nonzero()[0]
            for idx in list(target_nodes_indices):
                edge_list.append((v1, idx + v1 + 1, distances[idx]))
        return np.array(edge_list)

    # Initialises self to a new GIRG with the appropriate parameters using _fast_edge_list. The GIRG is stored in
    # adjacency matrix form in a csr_matrix, and in this paper all edge weights are 1.
    def sample_graph(self, edge_rvs=None):
        edge_list = self._fast_edge_list(self.vertex_set.locations,
                                         self.vertex_set.weights,
                                         self.vertex_set.torus_width,
                                         self.c1, self.c2, 1/(self.tau-1), self.alpha, self.sigma,
                                         on_torus=self.on_torus,
                                         edge_rvs=edge_rvs)
        distance_matrix = csr_matrix((edge_list[:,2], (edge_list[:,0], edge_list[:,1])),
                                     shape=(self.vertex_set.number_of_vertices,
                                            self.vertex_set.number_of_vertices))
        self._distance_matrix = distance_matrix + distance_matrix.T
    
    
# We use this class for geometric inhomogeneous random graphs.
class GIRG(InterpolatingKSRG):
    def __init__(self,
                 vertex_set,
                 alpha,
                 tau,
                 c1=1,
                 c2=1,
                 **kwargs):
        super().__init__(vertex_set,
                         alpha=alpha, 
                         tau=tau,
                         sigma=1,
                         c1=c1,
                         c2=c2,
                         **kwargs)
        

# We don't use this graph model in this paper.
class LongRangePercolation(SpatialGraph):
    def __init__(self,
                 vertex_set,
                 alpha,
                 c1=1,
                 c2=1,
                 **kwargs):
        super().__init__(vertex_set,
                         **kwargs)
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.parameters = ["alpha", "c1", "c2"]

    @staticmethod
    @jit(nopython=True)
    def _fast_edge_list(vertex_locations,
                        torus_width,
                        c1, c2, alpha,
                        on_torus,
                        edge_rvs=None):
        nr_of_vertices, dimension = vertex_locations.shape
        edge_list = []
        for v1 in prange(nr_of_vertices-1):
            other_locations = vertex_locations[v1+1:]
            distances = distance_from(other_locations,
                                      vertex_locations[v1],
                                      torus_width,
                                      on_torus)
            prob = c1*np.minimum(1, c2/(distances**dimension)**alpha)
            if edge_rvs is None:
                edge_rvs_v1 = np.random.random(len(distances))
            else:
                edge_rvs_v1 = edge_rvs[v1, v1+1:]
            target_nodes_indices = (edge_rvs_v1 < prob).nonzero()[0]
            for idx in list(target_nodes_indices):
                edge_list.append((v1, idx + v1 + 1, distances[idx]))
        return np.array(edge_list)

    def sample_graph(self, edge_rvs=None):
        edge_list = self._fast_edge_list(self.vertex_set.locations,
                                         self.vertex_set.torus_width,
                                         self.c1, self.c2, self.alpha,
                                         on_torus=self.on_torus,
                                         edge_rvs=edge_rvs)
        distance_matrix = csr_matrix((edge_list[:,2], (edge_list[:,0], edge_list[:,1])),
                                     shape=(self.vertex_set.number_of_vertices,
                                            self.vertex_set.number_of_vertices))
        self._distance_matrix = distance_matrix + distance_matrix.T


# We use this class for configuration model random graphs.
class ConfigurationModel(SpatialGraph):
    def __init__(self,
                 vertex_set,
                 distribution_type,
                 distribution_parameter,
                 mean_degree,
                 **kwargs):
        super().__init__(vertex_set,
                         **kwargs)
        self.distribution_type = distribution_type
        self.distribution_parameter = distribution_parameter
        self.mean_degree = mean_degree
        self.parameters = ["distribution_type", "distribution_parameter", "mean_degree"]

    # Initialises self to a new configuration model graph with the appropriate parameters. The graph is stored in
    # adjacency matrix form in a csr_matrix, and in this paper all edge weights are 1.
    def sample_graph(self):
        if self.distribution_type == "power_law":
            tau = self.distribution_parameter
            power_law_weights = self.vertex_set.weights**(-1/(tau-1))
            degrees = np.round(power_law_weights * self.mean_degree / power_law_weights.mean()).astype(np.int)
        elif self.distribution_type == "poisson":
            degrees = np.random.poisson(self.mean_degree, self.vertex_set.number_of_vertices)
        if degrees.sum() % 2 == 1:
            degrees[-1] += 1
        half_edges = np.array([i for i, degree in enumerate(degrees) for _ in range(degree)])
        np.random.shuffle(half_edges)
        edges = half_edges.reshape((half_edges.shape[0] // 2, 2))
        nonzero_sources, nonzero_targets = edges.T
        data = np.ones(len(nonzero_sources))
        sparse_matrix = csr_matrix((data, (nonzero_sources, nonzero_targets)),
                                   shape=(self.vertex_set.number_of_vertices, self.vertex_set.number_of_vertices))
        distance_matrix = ((sparse_matrix + sparse_matrix.T) > 0).astype(np.int)
        distance_matrix.setdiag(np.zeros(self.vertex_set.number_of_vertices))
        self._distance_matrix = distance_matrix
