from typing import NamedTuple, Union, Iterable, Any, Dict, Tuple, List, Callable, Optional
import numpy as np
import bisect


class Edge(NamedTuple):
    """
    A namedtuple. Represents an edge in a graph, specifically a StorageGraph. Contains 2 entries:
     - node1: The 1st node the edge is connected to.
     - node2: The 2nd node the edge is connected to.

    As a named tuple, it also supports all of the operations of a regular tuple.

    This is a bi-directional edge, so swapping the order of the edges in the tuple does nothing, it will still hash to
    the same value, and edges with their nodes swapped will equal each other. (Edge(a, b) == Edge(b, a))
    """
    node1: Union[str, int]
    node2: Union[str, int]

    def __hash__(self):
        """
        Custom Hash: Uses unordered hashing, so that Edge(a, b) and Edge(b, a) hash to the same value.
        """
        return hash(frozenset(self))

    def __eq__(self, other):
        """
        Uses unordered equivalence, so that Edge(a, b) and Edge(b, a) are considered equivalent.
        """
        if(isinstance(other, (Edge, Iterable))):
            return frozenset(self) == frozenset(other)

        return NamedTuple.__eq__(self, other)

"""
An edge-like object, used as argument for looking up an edge in a StorageGraph. Can be an Edge namedtuple object, or 
a tuple with 2 integers or strings, or combinations of both.
"""
EdgeLike = Union[Edge, Tuple[Union[int, str], Union[int, str]]]


class InvalidEdgeError(ValueError):
    pass


class InvalidNodeError(ValueError):
    pass


class StorageGraph:
    __slots__ = ["_node_names", "_name_to_idx", "_connections"]

    def __init__(self, node_names: Iterable[str]):
        """
        Create a new StorageGraph.

        :param node_names: An iterable of strings, the names of the nodes to store in this graph...
        """
        self._node_names = list(node_names)
        self._name_to_idx: Dict[str, int] = {name: i for (i, name) in enumerate(self._node_names)}
        self._connections = [{} for __ in range(len(self._name_to_idx))]

    def __len__(self):
        """
        Get the number of nodes in the graph.

        :returns: An integer, the number of nodes stored in this graph.
        """
        return len(self._connections)

    def _validate_index(self, idx: Union[int, str]) -> int:
        """
        PRIVATE: Validates an index.

        :param idx: An integer or string, referencing a node by either index or name.

        :returns: An integer, the index of the node, if valid.

        :throw InvalidNodeError: If the passed node reference is invalid or out of bounds...
        """
        idx = idx if(isinstance(idx, int)) else self._name_to_idx.get(idx, -1)

        if(not (0 <= idx < len(self._connections))):
            raise InvalidNodeError("Not a valid node!")

        return idx

    def _validate(self, edge: EdgeLike) -> Edge:
        """
        PRIVATE: Validate an edge.

        :param edge: An edge-like argument as received from the user.

        :returns: An sanitized Edge with 2 integers as it's node identifiers, being the indexes of the nodes.

        :throws InvalidEdgeError: If the provided edge has invalid node identifiers passed or connects a node to itself.
        """
        # Remove strings...
        cleaned_edge = Edge(*[(node if(isinstance(node, int)) else self._name_to_idx.get(node, -1)) for node in edge])

        if(not all(0 <= node < len(self._connections) for node in cleaned_edge)):
            raise InvalidEdgeError(f"The edge {edge} is not a valid edge for this graph!")

        if(cleaned_edge.node1 == cleaned_edge.node2):
            raise InvalidEdgeError(f"Can't connect node {cleaned_edge.node1} to itself!")

        return cleaned_edge

    def __getitem__(self, idx: Union[str, int, EdgeLike]) -> Union[Any, Iterable[Tuple[int, Any]]]:
        if(isinstance(idx, (Edge, tuple))):
            idx = self._validate(idx)
            return self._connections[idx.node1][idx.node2]
        else:
            idx = self._validate_index(idx)
            return self._connections[idx].items()

    def __contains__(self, idx: Union[str, int, EdgeLike]) -> bool:
        if (isinstance(idx, (Edge, tuple))):
            try:
                idx = self._validate(idx)
            except InvalidEdgeError:
                return False
            return (idx.node1 in self._connections) and ((idx.node2 in self._connections[idx.node1]))
        else:
            try:
                self._validate_index(idx)
                return True
            except InvalidNodeError:
                return False

    def __setitem__(self, edge: EdgeLike, value: Any):
        edge = self._validate(edge)

        self._connections[edge.node1][edge.node2] = value
        self._connections[edge.node2][edge.node1] = value

    def __delitem__(self, edge: EdgeLike):
        edge = self._validate(edge)

        del self._connections[edge.node1][edge.node2]
        del self._connections[edge.node2][edge.node1]

    def pop(self, edge: EdgeLike) -> Any:
        edge = self._validate(edge)

        result = self.__getitem__(edge)
        self.__delitem__(edge)

        return result

    def __iter__(self) -> Iterable[Edge]:
        """
        Iterate the edges of this graph.

        :returns: An iterable, containing Edge named tuples, the edges of this graph...
        """
        visited_edges = set()

        for n1, edge_lst in enumerate(self._connections):
            for n2 in edge_lst:
                edge = Edge(n1, n2)
                if(edge not in visited_edges):
                    visited_edges.add(edge)
                    yield edge

    def dfs(self, traversal_function: Optional[Callable[[Edge, Any], None]] = None) -> List[int]:
        """
        Run a depth-first-search of the entire graph, also returning a connected component list...

        :param traversal_function: An optional function to be run for each edge that gets traversed while performing
                                   DFS. Accepts 2 arguments:
                                    - An Edge tuple, the first value will be the node DFS just
                                      came from, the second value will be the node the DFS is currently on.
                                    - A value, being the value stored in the graph at the current node.

        :returns: A list of integers, representing the connected nodes/components they are a part of. All nodes as
                  indexes of the list that are a part of the same connected component will have the same integer id as
                  there value in the list.
        """
        traversal_function = traversal_function if(traversal_function is not None) else lambda a, b: None
        visited = [-1] * len(self)

        for n_i in range(len(self)):
            self._dfs_helper(traversal_function, visited, n_i, None)

        return visited

    def _dfs_helper(
        self,
        traversal_function: Callable[[Edge, Any], None],
        visited: List[int],
        current_node: int,
        prior_node: Optional[int] = None
    ):
        """
        PRIVATE: Helper to dfs method. Performs a dfs search recursively, resolving components...
        """
        # If already visited, return...
        if(visited[current_node] != -1):
            return

        # Stash component this is a part of in visited... (We use the first node of the component we visit as the
        # 'root' of that component.
        visited[current_node] = visited[prior_node] if(prior_node is not None) else current_node
        # If we have an actual prior node, run the edge traversal function.
        if(prior_node is not None):
            traversal_function(Edge(prior_node, current_node), self[prior_node, current_node])

        # Recurse: Run DFS on things connected to us...
        for other_n_i in self._connections[current_node]:
            self._dfs_helper(traversal_function, visited, other_n_i, current_node)

    def edges(self) -> Iterable[Edge]:
        """
        Iterate the edges of this graph.

        :returns: An iterable, containing Edge named tuples, the edges of this graph...
        """
        return iter(self)

    def values(self) -> Iterable[Any]:
        """
        Iterate the values of this graph.

        :returns: An iterable of anything, whatever values have been attached to the edges.
        """
        return (self._connections[edge.node1][edge.node2] for edge in self)

    def items(self) -> Iterable[Tuple[Edge, Any]]:
        """
        Get an iterable of both edges and values for this graph...

        :returns: An iterable of tuples containing 2 items:
                    - An Edge: named tuple, the edge in the graph.
                    - Anything, the value at the above given edge in the graph
        """
        return ((edge, self._connections[edge.node1][edge.node2]) for edge in self)

    def name_to_index(self, idx: Union[str, int]) -> int:
        """
        Convert a node name string to its integer index in the graph.

        :param idx: A string, being the node name. Can be an integer, in which case the index is simply validated.

        :returns: An integer, the index or id of the node in the graph.
        """
        return self._validate_index(idx)

    def index_to_name(self, idx: Union[str, int]) -> str:
        """
        Convert a node integer index to it's string name.

        :param idx: An integer, the index or id of the node. Can also be a string, in which case the name is just
                    validated.

        :returns: A string, the string name of the node in the graph.
        """
        idx = self._validate_index(idx)
        return self._node_names[idx]

    def node_names(self) -> List[str]:
        """
        Get the names of all of the nodes, in order...

        :returns: A list of strings, the names of all the nodes, in order as in the graph.
        """
        return self._node_names

    def __tojson__(self):
        return {
            "nodes": list(self._node_names),
            "edges": list(self.items())
        }

    @classmethod
    def __fromjson__(cls, data):
        s = cls(data["nodes"])
        for edge, val in data["edges"]:
            s[edge] = val
        return s

    def __str__(self):
        return f"Skeleton({str(dict(self.items()))}"


class Histogram:
    __slots__ = ["_bins", "_bin_size", "_bin_offset"]

    def __init__(self, bin_size: float = 1, bin_offset: float = 0):
        self._bins: Dict[int, Tuple[int, float]] = {}
        self._bin_size = bin_size
        self._bin_offset = bin_offset

    def add(self, value: float, weight: int = 1):
        # Compute the bin the value falls into...
        val_bin = int((value - self._bin_offset) / self._bin_size)

        freq, avg = self._bins.get(val_bin, (0, 0.0))
        # Running Average formula, see notebook...
        self._bins[val_bin] = (freq + 1, avg * (freq / (freq + 1)) + value * (1 / (freq + 1)))

    def get_bin_for_value(self, value: float) -> Tuple[float, int, float]:
        val_bin = int((value - self._bin_offset) / self._bin_size)
        freq, avg = self._bins.get(val_bin, (0, value))
        return (val_bin * self._bin_size + self._bin_offset, freq, avg)

    def __iter__(self) -> Iterable[float]:
        return (b * self._bin_size + self._bin_offset for b in self._bins)

    def bins(self) -> Iterable[Tuple[float, Tuple[int, float]]]:
        return ((b * self._bin_size + self._bin_offset, (freq, avg)) for b, (freq, avg) in self._bins.items())

    def get_max(self) -> Tuple[float, int, float]:
        max_info = (0, 0, 0)

        for (b, (freq, avg)) in self.bins():
            if(max_info[1] < freq):
                if((max_info[1] == freq) and (max_info[0] < b)):
                    continue
                max_info = (b, freq, avg)

        return max_info

    def get_quantile(
        self,
        quant: float,
        start_bin: float = None
    ) -> Tuple[float, int, float]:
        ordered_bins = sorted(self)
        ordered_indexes = sorted(self._bins)

        start_idx = bisect.bisect_left(ordered_bins, start_bin) if(start_bin is not None) else 0

        full_list = [self._bins[int((bin_i - self._bin_offset) / self._bin_size)][0] for bin_i in ordered_bins]
        sub_list = full_list[start_idx:]

        bin_num = ordered_indexes[min(start_idx + bisect.bisect_right(np.cumsum(sub_list) / np.sum(full_list), quant), len(ordered_bins) - 1)]
        freq, avg = self._bins[bin_num]
        return (bin_num, freq, avg)

    def get_mean_and_std(self) -> Tuple[float, float]:
        # Weighted average of bins...
        avgs = np.array([avg for (b, (freq, avg)) in self.bins()])
        freqs = np.array([freq for (b, (freq, avg)) in self.bins()])

        mean = np.sum(avgs * freqs) / np.sum(freqs)
        std = np.sqrt(np.sum(freqs * (avgs - mean) ** 2) / np.sum(freqs))

        return (mean, std)

    def get_std_using_mean(self, mean: Union[float, float]) -> float:
        # Weighted average of bins...
        mean = mean
        avgs = np.array([avg for (b, (freq, avg)) in self.bins()])
        freqs = np.array([freq for (b, (freq, avg)) in self.bins()])

        std = np.sqrt(np.sum(freqs * (avgs - mean) ** 2) / np.sum(freqs))

        return std

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{type(self).__name__}({str(dict(sorted((k, v) for k, v in self.bins())))})"