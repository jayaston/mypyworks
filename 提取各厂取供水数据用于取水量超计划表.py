# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

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
         ['1009','04281','m']]
list3 = [
         ['1001','00718','m'],
         
         ['1002','00718','m'],
         
         ['1003','00718','m'],
         
         ['1004','00718','m'],
         
         ['1005','00718','m'],
         
         ['1007','00718','m'],
         
         ['1016','00718','m']                 
         ]

shuju_df = tjfx.TjfxData().getdata('20180101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE','QUOTA_NAME'],columns = ['GROUP_NAME'],values='QUOTA_VALUE')
print(test)
test.info()
#test.columns
#按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon=='08']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
test = test[['江村水厂','西村水厂','石门水厂','北部水厂', '南洲水厂','西洲水厂', '新塘水厂']].T

#test = test.resample("Y").sum()

try:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),r"输出\2019取供水监控数据.xlsx"))
    test.to_excel(path)    
except:
    test.to_excel(r'./输出/2019取供水监控数据.xlsx')
