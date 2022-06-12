# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import sys
import os
os.getcwd()
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx

#import re    
import datetime as dt

#耗水30976，稽查24299，每月发单23765，每月追收24293，每月剔除24294，两月发单23594，两月追收23595，两月剔除23596

list1 =  [#公司         
         ['00','05339','m'],#一般工业
         ['00','05337','m'],#居民    
         ['00','05338','m'],#行政
         ['00','31846','m'],#经营服务业
         ['00','31847','m'],#特种
         
                 
         #东山         
         ['01','05339','m'],#一般工业
         ['01','05337','m'],#居民    
         ['01','05338','m'],#行政
         ['01','31846','m'],#经营服务业
         ['01','31847','m'],#特种
         
         #越秀
         ['02','05339','m'],#一般工业
         ['02','05337','m'],#居民    
         ['02','05338','m'],#行政
         ['02','31846','m'],#经营服务业
         ['02','31847','m'],#特种
         
         #荔湾
         ['03','05339','m'],#一般工业
         ['03','05337','m'],#居民    
         ['03','05338','m'],#行政
         ['03','31846','m'],#经营服务业
         ['03','31847','m'],#特种
         
         #海珠
         ['04','05339','m'],#一般工业
         ['04','05337','m'],#居民    
         ['04','05338','m'],#行政
         ['04','31846','m'],#经营服务业
         ['04','31847','m'],#特种
         
         #芳村
         ['05','05339','m'],#一般工业
         ['05','05337','m'],#居民    
         ['05','05338','m'],#行政
         ['05','31846','m'],#经营服务业
         ['05','31847','m'],#特种
         
         #黄埔
         ['06','05339','m'],#一般工业
         ['06','05337','m'],#居民    
         ['06','05338','m'],#行政
         ['06','31846','m'],#经营服务业
         ['06','31847','m'],#特种
         
         #白云
         ['07','05339','m'],#一般工业
         ['07','05337','m'],#居民    
         ['07','05338','m'],#行政
         ['07','31846','m'],#经营服务业
         ['07','31847','m'],#特种
         
         #天河
         ['08','05339','m'],#一般工业
         ['08','05337','m'],#居民    
         ['08','05338','m'],#行政
         ['08','31846','m'],#经营服务业
         ['08','31847','m'],#特种
         
         ]


shuju_df = tjfx.TjfxData().getdata('20210101','20220531',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE','QUOTA_NAME'],columns = ['GROUP_NAME'],values='QUOTA_VALUE')
# list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
# test.columns = list1

test.replace(np.nan,0,inplace = True)

test.eval("""
           中区分公司 = 东山片+越秀片+荔湾片
            东区分公司  = 黄埔片+天河片
            南区分公司  = 海珠片+芳村片
            北区分公司  = 白云片 
           """,inplace=True)
test.rename(columns= {'广州自来水公司':'公司',},inplace=True)
test=test.reindex(columns =['中区分公司','东区分公司','南区分公司','北区分公司'])

test1=test.unstack(1)
test2 = test1.shift(periods=12, axis=0)

test2 = pd.concat([test1,test2],axis=1,keys=['当期','同期'])
test2 = test2['2022']

test2 = pd.melt(test2,ignore_index=False)
test2.reset_index(inplace=True)
test2.columns=['月份','当期同期','分公司','大行业','售水量']
test2['月份']=test2['月份'].dt.strftime("%Y年%m月")


test2.to_excel(r'C:\Users\XieJie\mypyworks\输出\2022年行业售水量.xlsx',index=False)




