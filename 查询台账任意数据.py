﻿# -*- coding: utf-8 -*-
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


list1 = [['00','12316','m'],#公司学校
         ['00','04861','m'],#公司学校用水
         ['00','05337','m'],#公司居民
         ['00','05339','m'],#公司一般工业
         ['00','05340','m']#公司经营服务（19年11前）         
         ]
shuju_df = tjfx.TjfxData().getdata('20191001','20200627',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
test.columns = list1


#print(test)
test.info()
test.plot()
#plt.axhline(y=583357,ls="--")#添加水平直线

##按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon<='10']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
#test = test.resample("Y").sum()
test['公司去年同期取供比'] = test['广州自来水公司']['取供比'] - test['广州自来水公司']['取供比'].diff(365) 
test['西江原水管理所去年同期取供比'] = test['西江原水管理所']['取供比'] - test['西江原水管理所']['取供比'].diff(365) 
test['西村水厂去年同期取供比'] = test['西村水厂']['取供比'] - test['西村水厂']['取供比'].diff(365)
test['石门水厂去年同期取供比'] = test['石门水厂']['取供比'] - test['石门水厂']['取供比'].diff(365)
test['北部水厂去年同期取供比'] = test['北部水厂']['取供比'] - test['北部水厂']['取供比'].diff(365)
test['新塘水厂去年同期取供比'] = test['新塘水厂']['取供比'] - test['新塘水厂']['取供比'].diff(365)
test['西洲水厂去年同期取供比'] = test['西洲水厂']['取供比'] - test['西洲水厂']['取供比'].diff(365)
test['南洲水厂去年同期取供比'] = test['南洲水厂']['取供比'] - test['南洲水厂']['取供比'].diff(365)
test.sort_values('QUOTA_DATE')
test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2015-2019售水.xlsx')
