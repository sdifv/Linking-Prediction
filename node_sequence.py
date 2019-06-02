# -*- coding: utf-8 -*-
import node2vec as n2v
import networkx as nx


#####load network
def read_graph(args):
    if args.weighted:   #带权值
        g = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())#DiGraph类提供了许多有向图
    else:
        g = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in g.edges():
            g[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:   #有向
        g = g.to_undirected()   #无向
    return g

####random walk
def node_walk(args):
    nx_g = read_graph(args)
    g = n2v.Graph(nx_g, args.directed, args.p, args.q)#args.directed bool型用来标识
    #（有向图，无向图)， args.p，args.q分别是参数p和q, 这一步是生成一个图对象
    g.preprocess_transition_probs()#生成每个节点的转移概率向量
    walks = g.simulate_walks(args.num_walks, args.walk_length)#随机游走
    # walks是随机游走生成的多个节点序列，被当做文本输入，调用Word2Vec模型，生成向量
    return walks