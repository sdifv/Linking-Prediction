# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import argparse
import random
from deepwalk import walks
from deepwalk import graph as graph
import node_sequence as ns
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd
import load_data as ld
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def parse_args():
    ##输入的命令
    ##包括输入的网络，输出的embedding结果

    parser = argparse.ArgumentParser(description="Run disease2vec.")
    parser.add_argument('--input', nargs='?', default='Data\\interactome.txt',
	                    help='Input graph path')

    parser.add_argument('--output', nargs='?', default='Data\\gene.emb',
	                    help='Embeddings path')

    ##选择哪种方式来进行embedding，默认是node2vec，需要采用别的方法，请自行在空白处填写代码
    parser.add_argument('--methods', nargs='?', default='node2vec',
                        help='default is node2vec, obtain the vector of each node in the network.')

    #node2vec和deepwalk补充的各种参数
    parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk_length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

    parser.add_argument('--num_walks', type=int, default=1,
	                    help='Number of walks per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)#不带权重

    parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(directed=False)#无向图

    ###训练分类模型时添加训练集，测试集
    parser.add_argument('--train_neg', nargs='?', default='Data\\train_neg.txt',
	                    help='neg_train')

    parser.add_argument('--test_pos', nargs='?', default='Data\\test.txt',
                        help='pos_test')

    parser.add_argument('--test_neg', nargs='?', default='Data\\test_neg.txt',
                        help='neg_test')

    ##node pair向量的拼接方法
    parser.add_argument('--oper_name', nargs='?', default='average',
                        help='oper name is average, Hadamard, L1 or L2')

    return parser.parse_args()

############################embedding####################
def n2v_learn_embeddings(args):
    node_walks = ns.node_walk(args)

    # node_walks是随机游走生成的多个节点序列，被当做文本输入，调用Word2Vec模型，生成向量

    node_walks = list(list(map(str, walk)) for walk in node_walks)
    ###生成节点的随机序列
    print('done random walk')
    all_node = set([node for walk in node_walks for node in walk])
    model = Word2Vec(node_walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    ###采用skip-gram模型训练，获取节点的embedding
    print('done node embedding')

def learn_embeddings(args):
    ############
    G = graph.load_edgelist(args.input, undirected=args.undirected)
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.num_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

    print("Training...")
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, hs=1,
                     workers=1)

    model.wv.save_word2vec_format(args.output)
    ###采用skip-gram模型训练，获取节点的embedding
    print('done node embedding')
    ####          此处填写代码                 ####

    #####################
    return

def node_distance(args):
    g = nx.Graph()

    def build_graph(path,g):
        for line in open(path, 'r'):
            line_data = line.strip().split('\t')
            node1 = line_data[0]
            node2 = line_data[1]
            g.add_edge(node1, node2)

    build_graph(args.input,g)

    ####加载样本和标签
    ##只需要计算测试集合，没有训练过程
    test_pos = ld.load_samples(args.test_pos)
    test_neg = ld.load_samples(args.test_neg)
    labels = [1] * len(test_pos) + [0] * len(test_neg)  #正，反例产生的二分序列
    test_pos.extend(test_neg) #样本

    ####提示：计算点之间的最短路径，使用nx.shortest_distance
    shortest_dis = []
    ############
    for e in test_pos:
        dis=nx.shortest_path_length(g,e[0].__str__(), e[1].__str__())
        shortest_dis.append(-1*dis)
    


    ####             此处填写代码                   ###

    #####################
    # fpr,tpr,thresholds = metrics.roc_curve(labels,shortest_dis)
    # print(thresholds)
    
    ld.draw_ROC_curve(labels, shortest_dis)

def main(args):
    if args.methods == 'distance':
        node_distance(args)
    #用低维、稠密、实值的向量表示网络中的节点（含有语义关系，利于计算存储，不用再手动提特征（自适应性），
    # 且可以将异质信息投影到同一个低维空间中方便进行下游计算）
    
    if args.methods == 'node2vec':
        n2v_learn_embeddings(args)

        train_features, train_labels, test_features, test_labels = ld.load_feature(args)
        ###训练分类模型并预测缺失的边是否存在
        neigh = SVC(C=1.0, kernel='rbf', probability=True)
        # neigh = KNeighborsClassifier(n_neighbors=8)
        neigh.fit(train_features, train_labels)
        res = neigh.predict_proba(test_features)
        predict_res = list()
        for each in res:
            predict_res.append(each[1])
        ld.draw_ROC_curve(test_labels, predict_res)
        
    else:
        learn_embeddings(args)
        print("done1")
        train_features, train_labels, test_features, test_labels = ld.load_feature(args)
        ###训练分类模型并预测缺失的边是否存在
        print("done2")
        neigh = KNeighborsClassifier(n_neighbors=5)
        print("done3")
        neigh.fit(train_features, train_labels)
        print("done4")
        res = neigh.predict_proba(test_features)
        print("done5")
        '''
        neigh = SVC(C=1.0, kernel='rbf', gamma="auto",probability=True,tol=1)
        print("done3")
        neigh.fit(train_features, train_labels)
        print("done4")
        res = neigh.predict_proba(test_features)
        print("done5")
        '''
        predict_res = list()
        for each in res:
            predict_res.append(each[1])
        print("done6")
        ld.draw_ROC_curve(test_labels, predict_res)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

