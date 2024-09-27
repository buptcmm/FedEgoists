from typing import List
from typing import Dict
import copy
from docplex.mp.model import Model
import matplotlib as mpl
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
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

def Generate_competetive_graph(Compet_ratio, usr_num):
    Compet_Matrix = np.zeros([usr_num, usr_num])
    for i in range(usr_num):
        for j in range(i + 1, usr_num):
            p = random.random()
            if p < Compet_ratio:
                Compet_Matrix[i, j] = 1
                Compet_Matrix[j, i] = 1

    return Compet_Matrix

def MIQP_solver(path,no_ma,Benefit_Matrix,T):
    len_n = len(no_ma)

    mdl = Model('minimize_function')

    x = mdl.binary_var_list(len_n, name='x') #no_ma 中的点是否被选择 0,1

    #计算余弦相似度
    cosine_similarities = np.array([[np.dot(Benefit_Matrix[i], Benefit_Matrix[j]) /
                                     (np.linalg.norm(Benefit_Matrix[i]) * np.linalg.norm(Benefit_Matrix[j]))
                                     for j in range(len(Benefit_Matrix))]
                                    for i in range(len(Benefit_Matrix))])

    lamda = 0.1

    objective = -mdl.sum(x) - lamda * mdl.sum(x[i] * cosine_similarities[path[no_ma[i]]][path[no_ma[i]+1]] for i in range(len_n-1))

    mdl.minimize(objective)

    # 添加约束条件
    mdl.add_constraint(mdl.sum(x) <= T)
    # 求解问题
    solution = mdl.solve()
    solutions = []
    # 输出结果
    if solution:
        for i in range(len_n):
            #print(f"x{i + 1} = {x[i].solution_value}")
            if (x[i].solution_value == 1):
                solutions.append(i)
    else:
        print("No solution found")

    print(solutions)
    return solutions


def join_vector(Contribute_Matrix,Compet_Matrix):
    G_u = nx.from_numpy_matrix(np.array(Contribute_Matrix), create_using=nx.DiGraph)
    for i in range(len(Compet_Matrix)):
        for j in range(len(Compet_Matrix)):
            if(Compet_Matrix[i][j]==1):

                if(nx.has_path(G_u,i,j)):
                    return 0
    return  1


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
def generate_Sequence(usr_num,Benefit_Matrix):
    Contribution = np.sum((np.ones([usr_num,usr_num])-np.eye(usr_num))*Benefit_Matrix,axis=0)
    S = np.argsort(-Contribution)
    return S

def ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix, Compet_Matrix):
    B_i_value = []
    B_i_tmp = []
    vi = usr_i
    Contribute_Matrix_tmp = copy.deepcopy(Contribute_Matrix)
    G_u = nx.from_numpy_matrix(Contribute_Matrix_tmp, create_using=nx.DiGraph)

    for j in range(len(Benefit_Matrix[vi])):
        if(Benefit_Matrix[vi][j] > 0 and j != vi):
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
            if nx.has_path(G_u,usage_to_vj,vj):
                V_j.append(usage_to_vj)

        for client in V_j:
            for idx in range(len(Compet_Matrix[client])):
                if  Compet_Matrix[client][idx] == 1 and idx not in S_j:
                    S_j.append(idx)

        for client in S_j:
            if nx.has_path(G_u,vi,client):
                S_ij_1.append(client)

        for usage_from_vi in range(usr_num):
            if nx.has_path(G_u, vi, usage_from_vi):
                V_i.append(usage_from_vi)

        for client in V_i:
            for idx in range(len(Compet_Matrix[client])):
                if Compet_Matrix[client][idx] == 1 and idx not in S_i:
                    S_i.append(idx)

        for client in S_i:
            if nx.has_path(G_u,client,vj):
                S_ij_2.append(client)


        if len(S_ij_1) == 0 and len(S_ij_2) == 0:
            Contribute_Matrix_tmp[vj,vi] = 1
        else:
            Contribute_Matrix_tmp[vj,vi] = 0

    return Contribute_Matrix_tmp

def Greedy_allocation(usr_num,Compet_Matrix,Benefit_Matrix):
    Sequence = generate_Sequence(usr_num,Benefit_Matrix)
    Contribute_Matrix = np.diag([1]*usr_num)

    for usr_i in Sequence:
        Contribute_Matrix = ILP_solver(usr_num, usr_i, Contribute_Matrix, Benefit_Matrix, Compet_Matrix)

    Contribute_Matrix = Contribute_Matrix.T
    return Contribute_Matrix

if __name__ == '__main__':


    # Compet_Matrix = []
    C=[]
    Compet_Matrix = Generate_competetive_graph(0.3,6)
    print(Compet_Matrix)
    C = [[0, 0, 0, 1, 1, 0],
         [0, 0, 0,0,0, 0,],
         [0, 0, 0, 0, 0,0,],
         [1,0,0, 0,0,0,],
         [1,0, 0,0,0,0],
         [0, 0,0,0,0,0]]
    ajadcent_Benefit_Matrix = [[0,1,1,0,0,0],
                      [1,0,1,0,0,0],
                      [1,1,0,1,0,0],
                      [0,0,0,0,1,1],
                      [0,0,0,1,0,1],
                      [0,0,0,1,1,0]
                      ]
    # cnt = 0
    Benifit_Matrix = [
        [0.28580335, 0.2656967, 0.25705233, 0.06381585, 0.06381585, 0.06381585],
        [0.2797677, 0.26867577, 0.257764, 0.06459752, 0.06459752, 0.06459752],
        [0.27320266, 0.25010574, 0.28392816, 0.06425449, 0.06425449, 0.06425449],
        [0.06390542, 0.06390542, 0.06390542, 0.29843998, 0.2504042, 0.2594396],
        [0.06450091, 0.06450091, 0.06450091, 0.2659871, 0.2930896, 0.24742061],
        [0.0644559, 0.0644559, 0.0644559, 0.26602146, 0.2421869, 0.29842404]
    ]
    Benifit_Matrix1 = [
        [0.28580335, 0.2656967, 0.25705233, 0, 0, 0],
        [0.2797677, 0.26867577, 0.257764, 0, 0, 0],
        [0.27320266, 0.25010574, 0.28392816, 0, 0, 0],
        [0, 0, 0, 0.29843998, 0.2504042, 0.2594396],
        [0,0, 0, 0.2659871, 0.2930896, 0.24742061],
        [0, 0, 0, 0.26602146, 0.2421869, 0.29842404]
    ]
    # # for i in range(1):
    Contribute_Matrix1 = Greedy_allocation(6,C,Benifit_Matrix1)






