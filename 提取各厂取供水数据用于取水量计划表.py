﻿# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
#
import sys
sys.path.append(r'E:\pyworks\StatLedger\module')
import pandas as pd
import numpy as np
import shujuyuan as sj
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

shuju_df = sj.Datataizhang().getdata('20190101','20190831',list3)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)

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
test.to_excel(r'E:\pyworks\输出\2019取供水监控数据.xlsx')
