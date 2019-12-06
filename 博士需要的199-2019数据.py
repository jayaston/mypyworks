# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""

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


list1 = [['00','04660','m'],         
         ['00','04847','m'],
         ['00','00718','m']
         
         ]
shuju_df = tjfx.TjfxData().getdata('19990101','20191009',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
test.info()

##按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon<='10']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
#test = test.resample("Y").sum()

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\1999-2019博士需要售水数据1.xlsx')
