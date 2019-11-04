# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:32:20 2019

@author: XieJie
"""

import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    
import datetime as dt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt


list1 = [
         ['1001','04281','m'],
         
         ['1002','04281','m'],
         
         ['1003','04281','m'],
         
         ['1004','04281','m'],
        
         ['1005','04281','m'],
         
         ['1007','04281','m'],
         
         ['1016','04281','m']
                        
         ]
list2 = [      
         ['0023','00718','d']]
list3 = [
         
         
         ['1002','00718','m'],
         
         ['1003','00718','m']
         
                       
         ]

shuju_df = tjfx.TjfxData().getdata('20160101','20191031',list2)



shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')

#print(test)
#test.info()
test.columns=['花都供水总量']
test.sort_values('花都供水总量')
#按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon=='08']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['江村水厂','西村水厂','石门水厂','北部水厂', '南洲水厂','西洲水厂', '新塘水厂']].T

test = test.resample("Y").sum()
data = test['2018'].resample("m").sum()
try:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),r"输出\20191101何总需要供水数据5.xlsx"))
    test.to_excel(path)    
except:
    test.to_excel(r'./mypyworks/输出/20191101何总需要供水数据5.xlsx')
