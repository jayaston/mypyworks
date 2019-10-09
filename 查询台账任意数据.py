# -*- coding: utf-8 -*-
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


list1 = [['00','11930','d'],
         ['00','00718','d'],
         ['1009','11930','d'],
         ['1001','04281','d'],
         ['1001','00718','d'],
         ['1002','04281','d'],
         ['1002','00718','d'],
         ['1003','04281','d'],
         ['1003','00718','d'],
         ['1016','00718','d'],
         ['1004','04281','d'],
         ['1004','00718','d']                 
         ]
shuju_df = sj.Datataizhang().getdata('20160101','20190829',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
print(test)
test.info()

#按月份删选
test['mon'] = test.index.strftime('%m')
test = test[test.mon=='08']
test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
#test = test.resample("Y").sum()
test.to_excel(r'E:\pyworks\输出\2016-2018年8月取供比相关数据.xlsx')
