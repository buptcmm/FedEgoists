import copy
from typing import List
from typing import Dict
import copy
import cplex
import numpy as np
import random
import networkx as nx
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, id: int, parents: List[int], descendants: List[int]) -> None:
        """
        node initialise

        :param id:  node ID
        :param parents:  from which nodes can come to current node directly
        :param descendants:  from current node can go to which nodes directly
        """

        self.id = id
        self.parents = parents
        self.descendants = descendants


class Tarjan(object):
    """
    Tarjan's algorithm
    """

    def __init__(self, nodes: Dict[int, Node]) -> None:
        """
        data initialise

        :param nodes:  node dictionary
        """

        self.nodes = nodes

        # intermediate data
        self.unvisited_flag = -1
        self.serial = 0  # serial number of current node
        self.num_scc = 0  # current SCC
        self.serials = {i: self.unvisited_flag for i in nodes.keys()}  # each node's serial number
        self.low = {i: 0 for i in nodes.keys()}  # each node's low-link value
        self.stack = []  # node stack
        self.on_stack = {i: False for i in nodes.keys()}  # if each node on stack

        # run algorithm
        self.list_scc = []  # final result
        self._find_scc()

    def _find_scc(self):
        """
        algorithm main function
        """

        for i in self.nodes.keys():
            self.serials[i] = self.unvisited_flag

        for i in self.nodes.keys():
            if self.serials[i] == self.unvisited_flag:
                self._dfs(node_id_at=i)

        # result process
        dict_scc = {}
        for i in self.low.keys():
            if self.low[i] not in dict_scc.keys():
                dict_scc[self.low[i]] = [i]
            else:
                dict_scc[self.low[i]].append(i)
        self.list_scc = list(dict_scc.values())

    def _dfs(self, node_id_at: int):
        """
        algorithm recursion function

        :param node_id_at:  current node ID
        """

        self.stack.append(node_id_at)
        self.on_stack[node_id_at] = True
        self.serials[node_id_at] = self.low[node_id_at] = self.serial
        self.serial += 1

        # visit all neighbours
        for node_id_to in self.nodes[node_id_at].descendants:
            if self.serials[node_id_to] == self.unvisited_flag:
                self._dfs(node_id_at=node_id_to)

            # minimise the low-link number
            if self.on_stack[node_id_to]:
                self.low[node_id_at] = min(self.low[node_id_at], self.low[node_id_to])

        # After visited all neighbours, if reach start node of current SCC, empty stack until back to start node.
        if self.serials[node_id_at] == self.low[node_id_at]:
            node_id = self.stack.pop()
            self.on_stack[node_id] = False
            self.low[node_id] = self.serials[node_id_at]
            while node_id != node_id_at:
                node_id = self.stack.pop()
                self.on_stack[node_id] = False
                self.low[node_id] = self.serials[node_id_at]

            self.num_scc += 1


def Generate_competetive_graph(Compet_ratio, usr_num):
    Compet_Matrix = np.zeros([usr_num, usr_num])
    for i in range(usr_num):
        for j in range(i + 1, usr_num):
            p = random.random()
            if p < Compet_ratio:
                Compet_Matrix[i, j] = 1
                Compet_Matrix[j, i] = 1

    return Compet_Matrix



def bron_kerbosch(G):
    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = [None]

    subg = set(G)
    cand = set(G)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass




def generate_group(args,Compet_Matrix):
    negative_compete_matrix = np.zeros([len(Compet_Matrix), len(Compet_Matrix)])
    n = len(Compet_Matrix)
    coalition_group = []

    for i in range(len(Compet_Matrix)):
        for j in range(len(Compet_Matrix)):
            if (Compet_Matrix[i, j] == 1):
                negative_compete_matrix[i, j] = 0
            elif (Compet_Matrix[i, j] == 0):
                negative_compete_matrix[i, j] = 1

    negative_compete_graph = nx.from_numpy_matrix(negative_compete_matrix)
    negative_compete_graph_tmp = copy.deepcopy(negative_compete_graph)

    cliques = bron_kerbosch(negative_compete_graph_tmp)  # 寻找其中的团
    cliques_list = list(cliques)
    f = 0
    while len(cliques_list) > 0:
        cliques_list.sort(key=len, reverse=True)
        coalition_group.append(cliques_list[f])
        [negative_compete_graph_tmp.remove_node(nd) for nd in cliques_list[f]]
        cliques_list = list(nx.find_cliques(negative_compete_graph_tmp))
        f = 0

    return coalition_group



def generate_coalition_group(args,Compet_Matrix):
    negative_compete_matrix = np.zeros([len(Compet_Matrix), len(Compet_Matrix)])
    coalition_group = []

    for i in range(len(Compet_Matrix)):
        for j in range(len(Compet_Matrix)):
            if (Compet_Matrix[i, j] == 1):
                negative_compete_matrix[i, j] = 0
            elif (Compet_Matrix[i, j] == 0):
                negative_compete_matrix[i, j] = 1

    negative_compete_graph = nx.from_numpy_matrix(negative_compete_matrix)
    negative_compete_graph_tmp = copy.deepcopy(negative_compete_graph)

    cliques = nx.find_cliques(negative_compete_graph_tmp)  # 寻找其中的团
    cliques_list = list(cliques)
    f= 0
    while len(cliques_list) > 0:
        cliques_list.sort(key=len, reverse=True)
        coalition_group.append(cliques_list[f])
        [negative_compete_graph_tmp.remove_node(nd) for nd in cliques_list[f]]
        cliques_list = list(nx.find_cliques(negative_compete_graph_tmp))
        f = 0

    return coalition_group


def generate_Sequence(usr_num, Benefit_Matrix):
    Contribution = np.sum((np.ones([usr_num, usr_num]) - np.eye(usr_num)) * Benefit_Matrix, axis=0)
    S = np.argsort(-Contribution)
    return S


def ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix,
               Compet_Matrix):
    B_i_value = []
    B_i_tmp = []
    vi = usr_i
    Contribute_Matrix_tmp = copy.deepcopy(Contribute_Matrix)
    G_u = nx.from_numpy_matrix(Contribute_Matrix_tmp, create_using=nx.DiGraph)

    for j in range(len(Benefit_Matrix[vi])):
        if (Benefit_Matrix[vi][j] > 0 and j != vi):
            B_i_tmp.append(j)
            B_i_value.append(Benefit_Matrix[vi][j])
    B_i_value_arr = np.array(B_i_value)
    B_i_tmp_arr = np.array(B_i_tmp)
    B_i = copy.deepcopy(B_i_tmp_arr[np.argsort(-B_i_value_arr)])

    for vj in B_i:
        V_j = []
        S_j = []
        S_ij_1 = []
        V_i = []
        S_i = []
        S_ij_2 = []

        G_u = nx.from_numpy_matrix(Contribute_Matrix_tmp, create_using=nx.DiGraph)
        for usage_to_vj in range(usr_num):
            if nx.has_path(G_u, usage_to_vj, vj):
                V_j.append(usage_to_vj)

        for client in V_j:
            for idx in range(len(Compet_Matrix[client])):
                if Compet_Matrix[client][idx] == 1 and idx not in S_j:
                    S_j.append(idx)

        for client in S_j:
            if nx.has_path(G_u, vi, client):
                S_ij_1.append(client)

        for usage_from_vi in range(usr_num):
            if nx.has_path(G_u, vi, usage_from_vi):
                V_i.append(usage_from_vi)

        for client in V_i:
            for idx in range(len(Compet_Matrix[client])):
                if Compet_Matrix[client][idx] == 1 and idx not in S_i:
                    S_i.append(idx)

        for client in S_i:
            if nx.has_path(G_u, client, vj):
                S_ij_2.append(client)

        if len(S_ij_1) == 0 and len(S_ij_2) == 0:
            Contribute_Matrix_tmp[vj, vi] = 1
        else:
            Contribute_Matrix_tmp[vj, vi] = 0

    return Contribute_Matrix_tmp


def Greedy_allocation(usr_num, Compet_Matrix, Benefit_Matrix):
    Sequence = generate_Sequence(usr_num, Benefit_Matrix)
    Contribute_Matrix = np.diag([1] * usr_num)

    for usr_i in Sequence:
        Contribute_Matrix = ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix, Compet_Matrix)

    Contribute_Matrix = Contribute_Matrix.T
    return Contribute_Matrix


def is_valid(graph, coalition, node, id_fri):
    for i in range(len(graph)):
        if graph[node, i] == 1 and coalition[i] == id_fri:
            return False
    return True


def dfs(graph, node, coalition, num_fri, solutions):
    if node == len(graph):
        if coalition[0] == 0:
            solutions.add(tuple(coalition))
        return

    for id_fri in range(num_fri):
        if is_valid(graph, coalition, node, id_fri):
            coalition[node] = id_fri
            dfs(graph, node + 1, coalition, num_fri, solutions)
            coalition[node] = -1






