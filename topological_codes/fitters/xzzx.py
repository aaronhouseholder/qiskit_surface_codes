import numpy as np
import math
from random import choices
from itertools import combinations, product
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Tuple, List, Dict, Optional, TypeVar, Union, cast
from .base import TopologicalGraphDecoder
from .visualization import VisualizationMixin
from .mwpm import MWPMDecodingMixin

TQubit = Tuple[float, float, float]  # (time,row,column) ==> (t,i,j)
TQubitLoc = Tuple[float, float]  # (row,column) ==> (i,j)



class XZZXGraphDecoderBase(TopologicalGraphDecoder[TQubit]):
    """
    Class to construct XZZX code on a rotated lattice. The current noise model is 
    one of Z-biased noise on data qubits only. 
    """
    def __init__(self, code_params: Dict):
        super().__init__(code_params)
        if "d" not in self.code_params or "T" not in self.code_params:
            raise ValueError("Please include d and T in code_params.")
        if (
            "p" not in self.code_params or
            "eta" not in self.code_params or
            "bias" not in self.code_params
        ):
            raise ValueError("Please specify error model for XZZX decoder.")

        self.px, self.py, self.pz = self._get_err_probs()
        self.virtual = self._specify_virtual()
        self.S["A"] = nx.Graph()
        self.S["B"] = nx.Graph()
        self._make_syndrome_graph()

        self.paulis = {
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
            "I": np.array([[1, 0], [0, 1]]),
        }


    def _get_err_probs(self) -> List[float]:
        # Currently consider biased noise model only for purpose of demonstration
        bias_type = self.code_params["bias"]
        p = self.code_params["p"]
        eta = self.code_params["eta"]

        if bias_type == "Z":
            pz = p
            if eta <= 0:
                px = py = 0
            else:
                px = py = p / eta
        elif bias_type == "X":
            px = p
            if eta <= 0:
                pz = py = 0
            else:
                pz = py = p / eta
        else:
            py = p
            if eta <= 0:
                px = pz = 0
            else:
                px = pz = p / eta

        if (
            (1-px)/2 <= pz + py or (1-pz)/2 <= px + py
        ):
            raise ValueError("The noise is too high, check your noise model.")
        
        return px, py, pz


    def _specify_virtual(self) -> Dict[str, List[TQubit]]:
        """
        Returns the virtual syndrome nodes for subgraph A and B separately in dictionary
        Returns: 
            virtual = {"A": [(-1, r, c), ... ], "B": [(-1, r, c), ...]}
        """
        virtual: Dict[str, List[TQubit]] = {}
        virtual["A"] = []
        virtual["B"] = []
        
        for i in range(0, self.code_params["d"], 2):
            virtual["A"].append((-1, -0.5, i - 0.5))
            virtual["A"].append((-1, self.code_params["d"] - 0.5, i + 0.5))

            virtual["B"].append((-1, i + 0.5, -0.5))
            virtual["B"].append((-1, i - 0.5, self.code_params["d"] - 0.5))

        return virtual


    def _valid_syndrome(self, node, subgraph) -> bool:
        """
        Supplementary function. Check whether a syndrome node is valid on our lattice.
        Args: 
        node ((r, c)): The node to be checked
        subgraph ("A" or "B"): Specifying which subgraph the node is supposed to be on
        Returns:
        Boolean
        """
        r, c = node[0], node[1]
        if subgraph == "A":
            if (
                r > 0 
                and r < self.code_params["d"] - 1 
                and c > -1 
                and c < self.code_params["d"]
            ):
                return True
            else: 
                return False
        elif subgraph == "B":
            if (
                c > 0 
                and c < self.code_params["d"] - 1 
                and r > -1 
                and r < self.code_params["d"]
            ):
                return True
            else: 
                return False
        else:
            raise ValueError("Please enter a valid syndrome_graph_key: A or B")
    

    def get_error_syndrome(self):
        """
        Populate error on each data qubits, propagate errors to neighboring syndomes.
        Returns the actual logical error and the syndrome flips for error correction. 
        Returns:
            err_syndrome: Dictionary containing the nodes for subgraph A and B separately
            xL (int): The actual total logical X flips 
            zL (int): The actual total logical Z flips
        """
        pz = self.pz
        px = self.px
        py = self.py

        dat_qubits = list(product(range(self.code_params["d"]), range(self.code_params["d"])))
        pos_A = nx.get_node_attributes(self.S["A"], "pos")
        pos_B = nx.get_node_attributes(self.S["B"], "pos")

        anc_error = {"A": {pos: 0 for pos in pos_A}, "B": {pos: 0 for pos in pos_B}}
        err_syndrome = {"A": {}, "B": {}}
        xL = zL = 0


        flipped = {"X": [], "Y": [], "Z": []}
        for pos in dat_qubits:
            
            # Determine error or not according to error probability
            error = choices(["I", "X", "Y", "Z"], weights = [1-px-py-pz, px, py, pz])[0]
            if error == "X":
                flipped["X"].append(pos)
            elif error == "Y":
                flipped["Y"].append(pos)
            elif error == "Z":
                flipped["Z"].append(pos)

            # Count actual logical error from data qubits
            if pos[0] == self.code_params["d"] - 1:
                if pos[1] % 2 == 0 and error == "Z":
                    xL += 1
                elif pos[1] % 2 != 0 and error == "X":
                    xL += 1
                if error == "Y":
                    xL += 1
            if pos[1] == self.code_params["d"] - 1:
                if pos[0] % 2 == 0 and error == "X":
                    zL += 1
                elif pos[0] % 2 != 0 and error == "Z":
                    zL += 1
                if error == "Y":
                    zL += 1       

            left_up = (0, pos[0]-0.5, pos[1]-0.5)
            right_down = (0, pos[0]+0.5, pos[1]+0.5)
            left_down = (0, pos[0]+0.5, pos[1]-0.5)
            right_up = (0, pos[0]-0.5, pos[1]+0.5)

            # Propagate error from qubit to ancilla
            if left_up in self.S["A"] and (error == "Z" or error == "Y"):
                anc_error["A"][left_up] += 1
            elif left_up in self.S["B"] and (error == "Z" or error == "Y"):
                anc_error["B"][left_up] += 1

            if right_down in self.S["A"] and (error == "Z" or error == "Y"):
                anc_error["A"][right_down] += 1
            elif right_down in self.S["B"] and (error == "Z" or error == "Y"):
                anc_error["B"][right_down] += 1

            if left_down in self.S["A"] and (error == "X" or error == "Y"):
                anc_error["A"][left_down] += 1
            elif left_down in self.S["B"] and (error == "X" or error == "Y"):
                anc_error["B"][left_down] += 1

            if right_up in self.S["A"] and (error == "X" or error == "Y"):
                anc_error["A"][right_up] += 1
            elif right_up in self.S["B"] and (error == "X" or error == "Y"):
                anc_error["B"][right_up] += 1

        # Determine error syndromes in subgraph A and B
        for subgraph in ["A", "B"]:
            err_syndrome[subgraph] = [
                pos
                for pos in anc_error[subgraph]
                if anc_error[subgraph][pos] % 2 != 0
            ]
        
        return err_syndrome, xL, zL, flipped


    def _make_syndrome_graph(self) -> None:
        """
        Supplementary function. Make a complete NetworkX graph of the lattice with node position
        and edge weight. For visualization purposes. 
        Returns:
        node_graph: The complete NetworkX graph ready for plotting
        """

        start_nodes = {"A": (0.5, 0.5), "B": (0.5, 1.5)}
        for subgraph in ["A", "B"]:
            for t in range(self.code_params["T"]):
                start_node = start_nodes[subgraph]
                self.S[subgraph].add_node(
                    (t,) + start_node,
                    virtual=0,
                    pos=(start_node[1], -start_node[0]),
                    time=t,
                    pos_3D=(
                        start_node[1],
                        -start_node[0],
                        t,
                    ),
                )
                self._populate_syndrome_graph(
                    subgraph, (t,) + start_nodes[subgraph], [], t
                )

            syndrome_nodes_t0 = [
                x 
                for x, y in self.S[subgraph].nodes(data=True)
                if y["time"] == 0
            ]
            for node in syndrome_nodes_t0:
                space_label = (node[1], node[2])
                for t in range(0, self.code_params["T"] - 1):
                    self.S[subgraph].add_edge(
                        (t,) + space_label, (t + 1,) + space_label, distance=1
                    )

        return


    def _populate_syndrome_graph(self, subgraph: str, curr_node: TQubit, visited_nodes: List[TQubit], t: int) -> None:
        """
        Recursive function to populate syndrome subgraph at time 0 with syndrome_key A/B. The current_node
        is connected to neighboring nodes without revisiting a node.

        Args:
            subgraph ("A" or "B"): Which A/B syndrome subgraph these nodes are from.
            current_node ((0, r, c)): Current syndrome node to be connected with neighboring nodes.
            visited_nodes ([(0, r, c),]): List of syndrome nodes which have already been traver.
            node_graph (dictionary of two nx graphs): For appending nodes and edges to the complete graph

        Returns:
            None: function is to traverse the syndrome nodes and connect neighbors
        """
        neighbors = []
        r, c = curr_node[1], curr_node[2]
        neighbors.append((r - 1, c - 1, "Z"))
        neighbors.append((r + 1, c + 1, "Z"))
        neighbors.append((r - 1, c + 1, "X"))
        neighbors.append((r + 1, c - 1, "X"))

        normal_neighbors = [
            n 
            for n in neighbors
            if self._valid_syndrome(n, subgraph)
            and (t, n[0], n[1]) not in visited_nodes
        ]

        virtual_neighbors = [
            n
            for n in neighbors
            if (-1, n[0], n[1]) in self.virtual[subgraph]
            and (-1, n[0], n[1]) not in visited_nodes
        ]

        if not normal_neighbors and not virtual_neighbors:
            return
        
        for target in normal_neighbors:
            target_node = (t,) + target[:-1]
            if not self.S[subgraph].has_node(target_node):
                self.S[subgraph].add_node(
                    target_node,
                    virtual=0,
                    pos=(target[1], -target[0]),
                    time=t,
                    pos_3D=(target[1], -target[0], t),
                )
            if target[2] == "Z":
                if self.pz == 0 and self.py == 0:
                    continue
                else:
                    weight = round(math.log((1- self.px - self.py - self.pz)/(self.pz + self.py)), 10)
            elif target[2] == "X":
                if self.px == 0 and self.py == 0:
                    continue
                else:
                    weight = round(math.log((1- self.px - self.py - self.pz)/(self.px + self.py)), 10)
            
            self.S[subgraph].add_edge(
                curr_node, target_node, distance=weight
            )

        for target in virtual_neighbors:
            target_node = (-1,) + target[:-1]
            if not self.S[subgraph].has_node(target_node):
                self.S[subgraph].add_node(
                    target_node,
                    virtual=1,
                    pos=(target[1], -target[0]),
                    time=-1,
                    pos_3D=(target[1], -target[0], (self.code_params["T"] - 1) / 2),
                )
            if target[2] == "Z":
                if self.pz == 0 and self.py == 0:
                    continue
                else:
                    weight = round(math.log((1- self.px - self.py - self.pz)/(self.pz + self.py)), 10)
            elif target[2] == "X":
                if self.px == 0 and self.py == 0:
                    continue
                else:
                    weight = round(math.log((1- self.px - self.py - self.pz)/(self.px + self.py)), 10)

            self.S[subgraph].add_edge(
                curr_node, target_node, distance=weight
            )

        visited_nodes.append(curr_node)

        for target in normal_neighbors:
            self._populate_syndrome_graph(
                subgraph, (t,) + target[:-1], visited_nodes, t
            )
        
        for target in virtual_neighbors:
            self._populate_syndrome_graph(
                subgraph, (-1,) + target[:-1], visited_nodes, t
            )    


    def _make_error_graph(self, err_syndrome, multi=False) -> Dict[str, nx.Graph]:
        """
        Construct error graph of subgraph A and B consisting of (node_a, node_n, distance)
        for MWPM. Multi-path summation optional. Distance calculated in self._path_summation(...),
        degeneracy calculated in self._degeneracy_cal.
        Args:
            err_syndrome: Dictionary containing the nodes for subgraph A and B separately
            multi: True for multi-path summation, False ignores such degeneracy
        Returns:
            error_graph: Dictionary containing the error graph ready for MWPM for subgraph A and B separately
        """
        error_graph = {"A": nx.Graph(), "B": nx.Graph()}
        for subgraph in ["A", "B"]:
            virtual_dict = nx.get_node_attributes(self.S[subgraph], "virtual")
            time_dict = nx.get_node_attributes(self.S[subgraph], "time")

            nodes = err_syndrome[subgraph] + self.virtual[subgraph] # All the nodes that can be matched
            if len(nodes) % 2 != 0:
                nodes.append((-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5)) # Total nodes that can be matched must be even
                self.virtual[subgraph].append((-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5))
                error_graph[subgraph].add_node(
                        (-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5),
                        virtual=1,
                        pos=(self.code_params["d"] + 0.5, -(self.code_params["d"] + 0.5)),
                        time=-1,
                        pos_3D=(self.code_params["d"] + 0.5, -(self.code_params["d"] + 0.5), (self.code_params["T"] - 1) / 2),
                    )


            for node in nodes:
                if node == (-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5):
                    continue
                if not error_graph[subgraph].has_node(node):
                    error_graph[subgraph].add_node(
                        node,
                        virtual=virtual_dict[node],
                        pos=(node[2], -node[1]),
                        time=time_dict[node],
                        pos_3D=(node[2], -node[1], time_dict[node]),
                    )

            for source, target in combinations(nodes, 2):
                if (
                    source in self.virtual[subgraph]
                    and target in self.virtual[subgraph]
                ):
                    distance = 0.0
                elif (
                    source == (-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5) 
                    or target == (-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5)
                ):
                    continue
                else:
                    distance = float(
                        nx.shortest_path_length(
                            self.S[subgraph], source, target, weight="distance"
                        )
                    )

                    if multi:
                        deg = self._path_degeneracy(source, target, subgraph)
                        distance -= math.log(deg)
                    
                    distance = - distance

                error_graph[subgraph].add_edge(source, target, weight=distance)

            if (-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5) in self.virtual[subgraph]:
                self.virtual[subgraph].remove((-1, self.code_params["d"] + 0.5, self.code_params["d"] + 0.5))
        
        return error_graph


    def _path_degeneracy(self, source: TQubit, target: TQubit, subgraph: str) -> int:
        """
        Calculates the distance between nodes. If multi-path summation is not implemented, 
        distance equals to edge number times edge weight (X/Z separately). If multi-path
        summation is implemented, degeneracy will be added as an extra term. 
        Args:
            source (tuple): The starting syndrome/virtual node
            target (tuple): The ending syndrome/virtual node
            subgraph ("A" or "B"): Specifying which subgraph the nodes are supposed to be on
        Returns:
            distance (float): distance between the source and target node for MWPM
        """
        shortest_paths = list(nx.all_shortest_paths(subgraph, source, target, weight="distance"))
        degeneracy = len(shortest_paths)

        # TODO OPTIMIZE: CAN AT LEAST CUT ITERATION BY PASSING AN EXTRA ARGUMENT OF MATCHED VIRTUAL AND SYNDROME PAIR
        # If one of the nodes is virtual, check degeneracy of the other with all virtual nodes
        match = None
        if source[0] < 0:
            match = target
            matched = source
        elif target[0] < 0:
            match = source
            matched = target
        
        if match:
            shortest_distance = nx.shortest_path_length(
                subgraph, source, target, weight="distance"
            )
            
            self.virtual[subgraph].remove(matched)
            for virtual in self.virtual[subgraph]:
                distance = nx.shortest_path_length(
                    subgraph, match, virtual, weight="distance"
                )
                if distance == shortest_distance:
                    degeneracy += len(
                        list(
                            nx.all_shortest_paths(
                                subgraph, match, virtual, weight="distance"
                            )
                        )
                    )
            self.virtual[subgraph].append(matched)

        return degeneracy


    def _run_mwpm(self, matching_graph: nx.Graph) -> List[Tuple[TQubit, TQubit]]:
        matches = nx.max_weight_matching(matching_graph, maxcardinality=True)
        filtered_matches =[
            (source, target)
            for (source, target) in matches
            if source[0] >=0  or target[0] >= 0 
        ]

        return filtered_matches


    def error_correct(self, matches):
        """
        Error correct according to syndromes, returned values are compared with
        actual logical error to determine the logical error rates. 
        Args:
            matches ([(node_a, node_b, edge), ...]): A list of all the matches from MWPM
        Retuns:
            xL (int): The calculated total logical X flips 
            zL (int): The calculated total logical Z flips
        """
        xL = zL = 0
        for match in matches:
            if match[0][0] < 0 and match[1][0] < 0:
                continue

            for node in match:
                if node in self.virtual["A"] and node[1] == self.code_params["d"] - 0.5:
                    xL += 1
                elif node in self.virtual["B"] and node[2] == self.code_params["d"] - 0.5:
                    zL += 1
            
        return xL, zL


def graph_2D(G, edge_label):
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, edge_label)
    labels = {x: round(y, 3) for (x, y) in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()