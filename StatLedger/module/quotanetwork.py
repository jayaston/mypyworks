# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:01:23 2019

@author: XieJie
"""
#import sys
#import os
#try:
#    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
#except:
#    sys.path.append(r'.\mypyworks\StatLedger\module')
import cumcalculate as cc
from tjfxdata import TjfxData
import re 
import networkx as nx
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
from graphviz import Digraph

##指标名称与指标编码对应字典
all_quota = TjfxData().get_all_quota()
all_quota_dict = dict(zip(all_quota['QUOTA_CODE'],all_quota['QUOTA_NAME']))
##部门编码与部门名称对应字典
all_dept = TjfxData().get_all_dept()
all_dept_dict = dict(zip(all_dept['GROUP_CODE'],all_dept['GROUP_NAME']))

#用networkx生成网络关系函数
def quota_network():
    gongshiku = cc.order_gongshiku()
    #编码格式的network
    gongshiku['network'] = gongshiku.apply(lambda x : [(j,i) for i in [x['zhibiao']] for j in x['zhibiaoji'] ],axis=1)
    networklist = [i for j in list(gongshiku['network']) for i in j]
    #用networkx生成网络关系
    G = nx.DiGraph()
    G.add_edges_from(networklist)
    #nx.draw(G,with_labels=True)
    return G



def edgepred(inputnode,prededgelist):    
    result = dict(quota_network().pred[inputnode])
    if result != {}:        
        for i in result.keys():         
            prededgelist.append((i,inputnode))
            edgepred(i,prededgelist)            
    else:
      pass
   
def subedge(quota):
    prededgelist = []
    edgepred(quota,prededgelist)
    return prededgelist



def subnetwork(quota,path): 
    Gsub = nx.DiGraph()    
    for i in subedge(quota):
        a = i[0]
        a_result = re.findall(r'\b[a-z]_(\d+)_(\d+)\b',a)
        a_result=all_dept_dict.get(a_result[0][0])+all_quota_dict.get(a_result[0][1])
        b= i[1]
        b_result = re.findall(r'\b[a-z]_(\d+)_(\d+)\b',b)
        b_result=all_dept_dict.get(b_result[0][0])+all_quota_dict.get(b_result[0][1])
        Gsub.add_edge(a_result,b_result)
    nx.draw(Gsub,with_labels=True)
    plt.savefig(path)




def subgraphviz(quota,path):
    network = []
    for i in subedge(quota):
        for j in i:
            network.append(j)
    networkset = list(set(network))
    
    prededgedict = {i:all_dept_dict.get(re.findall(r'\b[a-z]_(\d+)_(\d+)\b',i)[0][0])+all_quota_dict.get(re.findall(r'\b[a-z]_(\d+)_(\d+)\b',i)[0][1])  for i in networkset }
        
   
    #采用graphviz作图    
    dot = Digraph()
    for i in list(prededgedict.keys()):
        dot.node(name=i, label=prededgedict.get(i), fontname="Microsoft YaHei")
        
    for i in subedge(quota):
        dot.edge(i[0],i[1],fontname="Microsoft YaHei")
    
#    dot.view()
    dot.render(path, view=True)

if __name__ == "__main__" :  
    subgraphviz('d_1001_31195')
    # subgraphviz('d_1001_31195',r'E:\pyworks\输出\network-output\round-table.gv')
    # subnetwork('d_1001_31195',r'E:\pyworks\输出\network-output\test.png')

##文字格式network
#def zhibiao_codetoname(x):#对数字编码还原成名称
#            result = re.findall(r'\b[a-z]_(\d+)_(\d+)\b',x['zhibiao'])
#            result=all_dept_dict.get(result[0][0])+all_quota_dict.get(result[0][1])
#            return result
#        
#def zhibiaoji_codetoname(x):#对数字编码还原成名称
#            result =  [re.findall(r'\b[a-z]_(\d+)_(\d+)\b',i) for i in x['zhibiaoji']]
#            result= [all_dept_dict.get(j[0][0])+all_quota_dict.get(j[0][1]) for j in result]
#            return result    
###将指标编码还原成中文
#gongshiku['zhibiao'] = gongshiku.apply(zhibiao_codetoname,axis=1)
#gongshiku['zhibiaoji'] = gongshiku.apply(zhibiaoji_codetoname,axis=1)
#gongshiku['network'] = gongshiku.apply(lambda x : [(j,i) for i in [x['zhibiao']] for j in x['zhibiaoji'] ],axis=1)
#


#通过提供节点筛选子集
#def nodepred(inputnode,prednodelist) :
#    
#    result = dict(G.pred[inputnode])
#    if result != {}:
#        prednodelist.append(inputnode)
#        for i in result.keys():         
#            
#            nodepred(i,prednodelist)
#            
#    else:
#        prednodelist.append(inputnode)
#    return prednodelist
    

   
#sub_graph = G.subgraph(prednodelist)
#
#nx.draw(sub_graph,with_labels=True)


#count = 0
#nodenamelist = []
#for i in nodeslist:
#    count = count + 1    
#    nodenamelist.append(chr(ord('A') + count))
#    
#nodeslist =  list(set(list(gongshiku['zhibiao'])+[j for i in list(gongshiku['zhibiaoji']) for j in i]))
#  
#nodedict = dict(zip(nodeslist,nodenamelist))
#
#for key in nodedict:
#    print(nodedict[key]+','+key)    
#    dot.node(key)
#



#for i in networklist:
#    dot.edge(i[0],i[1])
#
#dot.view()
#dot.render('test-output/round-table.gv', view=True)


#    
#
#dot.node(name="A", label="老师", fontname="Microsoft YaHei")
#dot.node('B', '学生', fontname="Microsoft YaHei")
#dot.edge("A", "B", label="教学", fontname="Microsoft YaHei")





