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
         ['06','05339','m'],#一般工业
         ['06','05337','m'],#居民    
         ['06','05338','m'],#行政
         ['06','31846','m'],#经营服务业
         ['06','31847','m'],#特种
         
         ]

list2 = [['00','06720','h'],#  压力   
         ['1001','06720','h'],
         ['1002','06720','h'],
         ['1003','06720','h'],
         ['1016','06720','h'],
         ['1004','06720','h'],
         ['1005','06720','h'],
         ['1006','06720','h'],
         ['1007','06720','h'],
         ['00','05960','h'],#  供水量   
         ['1001','05960','h'],
         ['1002','05960','h'],
         ['1003','05960','h'],
         ['1016','05960','h'],
         ['1004','05960','h'],
         ['1005','05960','h'],
         ['1006','05960','h'],
         ['1007','05960','h'],           
         ]
shuju_df = tjfx.TjfxData().getdata('20000101','20221231',list2)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
test.columns = list1

test = test.resample('m').mean()

test.to_csv(r'C:\Users\XieJie\mypyworks\输出\2000-2022年时数据.csv')

#求最大日
list2 = [['00','12190','d'],#最高温度
         ['00','12225','d'],#最低温度
         ['00','12260','d'],#平均温度
         ['00','31196','d'],#最高湿度
         ['00','31197','d'],#最低湿度
         ['00','31198','d'],#最低湿度
         ['00','00718','d'],
         ['1001','00718','d'],
         ['1002','00718','d'],
         ['1003','00718','d'],
         ['1016','00718','d'],
         ['1004','00718','d'],
         ['1005','00718','d'],
         ['1006','00718','d'],
         ['1007','00718','d']         
         ]
shuju_df = tjfx.TjfxData().getdata('20190101','20191231',list2)
shuju_df.info()
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

# shuju_df2 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
#                           aggfunc=np.sum)
# shuju_df2.reset_index(inplace=True)
# shuju_df2.info()
#按年最大值
get_max = lambda x: x[x['QUOTA_VALUE']==x['QUOTA_VALUE'].max()][['QUOTA_DATE','QUOTA_VALUE']]
                        
# python就是灵活啊。
get_max.__name__ = "maxday"
test = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'GROUP_NAME']).apply(get_max)

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2019年各厂最高日供水总量.xlsx')
