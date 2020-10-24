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


list1 = [
         ['1007','04281','d'],
         ['1007','00718','d']
         
         ]
shuju_df = tjfx.TjfxData().getdata('20200901','20200930',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
test.columns = list1

#print(test)
test.to_excel(r'C:\Users\XieJie\mypyworks\输出\20201019吴部需要南洲取供比.xlsx')

test.info()

