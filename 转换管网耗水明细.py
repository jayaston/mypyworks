# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:01:47 2021

@author: XieJie
"""

import pandas as pd

print(pd.__version__)


df_2019=pd.read_excel(r'e:\专题工作（重要）\产销差工作\产销差数据与资料积累\2019耗水明细.xlsx',index_col=[0,1,2],header=[0,1])
df_2020=pd.read_excel(r'e:\专题工作（重要）\产销差工作\产销差数据与资料积累\2020年耗水明细.xlsx',index_col=[0,1,2],header=[0,1])
df = pd.concat([df_2019,df_2020],axis=1)
df.columns.names = ['年月', '分公司'] 
df.index.names = ['是否收费', '一级分类', '二级分类'] 
df1 = df.stack(level=0) 
df2 = df1.unstack(level=[0,1,2]) 
df3 = df2.reorder_levels([1,2,3,0], axis = 1)
df4 = df3.sort_index(axis=1,level=[0,1,2],ascending=True,sort_remaining=False)
df4.to_excel(r'C:\Users\XieJie\mypyworks\输出\2019-2020耗水明细水量.xlsx')
df