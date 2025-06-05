import pandas as pd
import numpy as np
import networkx as nx
import os

def loadData(path):

    node_dict = {}
    df = pd.read_csv(path, index_col=False, header=None)
    arr = df.values
    node_list = []
    for each in arr:
        node_list.extend(list(map(int, each[0].split(" "))))
    node_arr = np.unique(np.array(node_list))
    for i in range(0, len(node_arr)):
        node_dict[node_arr[i]] = i
    return node_dict, node_arr, node_list, arr

class HyperG:

    N = 0               #节点个数
    M = 0              #超边个数
    node_dict = {}    #节点：节点序号，0号节点是节点1，270号节点是节点272，没有271节点
    node_arr = []    #从大到小排序节点，无重复，可能有的节点并没有出现在网络中
    arr = []        #所读取网络数据的每一行 [['53 137 237 238']['235 243']]

    def setN(self, N):

        self.N = N

    def setM(self, M):

        self.M = M

    def getN(self):

        return self.N

    def getM(self):

        return self.M

    def initData(self, node_dict, node_arr, node_list, arr):

        self.node_dict = node_dict
        self.node_arr = node_arr
        self.arr = arr



    '''cpt是啥意思'''
    def cptSize(self):

        node_dict = self.node_dict
        node_arr = self.node_arr
        arr = self.arr
        self.N = len(list(node_arr))
        self.M = len(arr)
        return (self.N, self.M)

    def get_edge_dict(self, path):

        node_dict, node_arr, node_list, arr = loadData(path)
        hpe_dict = {}       #hpe_dict表示  所有 超边序号：超边所包含的所有节点序号 1:[0,2,3,4,12,43]
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        i = 0
        for each in arr:
            new_list = []
            nodes_index_list = list(map(int, each[0].split(" ")))
            for index in nodes_index_list:
                new_list.append(node_dict[index])
            hpe_dict[i] = new_list
            i = i + 1
        return hpe_dict

    def get_nodes_dict(self, path):

        nodes_dict = {}
        node_dict, node_arr, node_list, arr = loadData(path)
        total = len(node_dict.values())
        for i in range(0, total):
            nodes_dict[i] = []           #nodes_dict表示  所有 节点序号：节点所在超边序号 0:[4,12,13]
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        i = 0
        for each in arr:
            nodes_index_list = list(map(int, each[0].split(" ")))
            for index in nodes_index_list:
                nodes_dict[node_dict[index]].append(i)
            i = i + 1
        return nodes_dict

    def get_hyper_degree(self, path, type):

        if type == 'node':      #返回 节点序号：节点所在的超边数   点超度
            dict = HyperG.get_nodes_dict(self, path)
        elif type == 'edge':   #返回  超边序号：超边所包含节点数   超边基数
            dict = HyperG.get_edge_dict(self, path)
        for each in dict:
            dict[each] = len(dict[each])
        return dict

    def get_average_degree(self, path, type):    #平均点超度、超边基数

        deg_dict = HyperG.get_hyper_degree(self, path, type)
        deg_list = deg_dict.values()
        avg_deg = sum(deg_list) / len(deg_list)
        return avg_deg

    def get_adjacent_node(self, path):   #adjacent,相邻的，返回所有节点：[相邻节点]   但去重之后，就忽略了节点之间的关联强度

        dict_node = HyperG.get_nodes_dict(self, path)
        dict_edge = HyperG.get_edge_dict(self, path)

        adj_dict = {}
        for j in range(0, len(dict_node)):
            adj_dict[j] = []
        for i in dict_node:
            edge_list = dict_node[i]
            for edge in edge_list:
                adj_dict[i].extend(dict_edge[edge])
        for k in range(0, len(adj_dict)):
            adj_dict[k] = list(np.unique(np.array(adj_dict[k])))
            # 去掉自环，重边
            if k in adj_dict[k]:
                adj_dict[k].remove(k)
        return adj_dict

    def get_projected_network(self, path):

        adj_dict = HyperG.get_adjacent_node(self, path)
        G = nx.Graph()
        G.add_nodes_from(list(adj_dict.keys()))    #添加边[2,3,4,6]
        for from_node in adj_dict:
            node_list = adj_dict[from_node]
            for to_node in node_list:
                G.add_edge(from_node, to_node)        #完全图
        return G

    def get_clustering_coefficient(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.average_clustering(G)

    def get_average_neighbor_degree(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.average_neighbor_degree(G)

    def get_density(self, path):

        G = HyperG.get_projected_network(self, path)
        return nx.density(G)

    def get_average_shortest_path_length(self, path):

        path_lengths = []
        G = HyperG.get_projected_network(self, path)
        node_list = G.nodes
        for node in range(0, len(node_list)):
            path_value_list = list(nx.shortest_path_length(G, target=node).values())
            path_lengths.extend(path_value_list)
        return sum(path_lengths) / len(path_lengths)

    def get_diameter(self, path):

        path_lengths = []
        G = HyperG.get_projected_network(self, path)
        node_list = G.nodes
        for node in range(0, len(node_list)):
            path_value_list = list(nx.shortest_path_length(G, target=node).values())
            path_lengths.extend(path_value_list)
        return max(path_lengths)

    def get_average_adj_degree(self, path):
        adj_dict = HyperG.get_adjacent_node(self, path)
        sum = 0
        for i in adj_dict:
            sum = sum + len(adj_dict[i])
        return sum / len(adj_dict.keys())




def get_topo_hypergraph(path):
    informations = []
    # global HG
    
    HG = HyperG()
    node_dict, node_arr, node_list, arr = loadData(path)
    HG.initData(node_dict, node_arr, node_list, arr)
    size = HG.cptSize()
    m,n = size
    avg_node_deg = HG.get_average_adj_degree(path)
    avg_node_degree = HG.get_average_degree(path, 'node')
    avg_edge_degree = HG.get_average_degree(path, 'edge')
    clustering_coefficient = HG.get_clustering_coefficient(path)
    average_shortest_path_length = HG.get_average_shortest_path_length(path)
    diameter = HG.get_diameter(path)
    density = HG.get_density(path)
    informations.append(m)
    informations.append(n)
    informations.append(avg_node_deg)
    informations.append(avg_node_degree)
    informations.append(avg_edge_degree)
    informations.append(clustering_coefficient)
    informations.append(average_shortest_path_length)
    informations.append(diameter)
    informations.append(density)
    # print(informations)
    return np.round(informations, 2)

if __name__ == '__main__':
    paths = os.listdir('../0-hypergraph_datasets/Ori_data3/')
    values = ['n','m','deg','dH','dE','c','<d>','$','&']
    datasets = [i[:-4] for i in paths]
    result = pd.DataFrame(index = datasets, columns= values)
    
    for i, path in enumerate(paths):
        all_path = '../0-hypergraph_datasets/Ori_data3/' + path
        informations = get_topo_hypergraph(all_path)
        result.loc[datasets[i]] = informations
        
    result.to_excel('topo3.xlsx')