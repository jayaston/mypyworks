# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""

import sys
import os
#os.getcwd()
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
import bumendata as bm
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
         ['1016','04281','d'],
         ['1016','00718','d'],
         ['1004','04281','d'],
         ['1004','00718','d'],
         ['1005','04281','d'],
         ['1005','00718','d'],
         ['1007','04281','d'],
         ['1007','00718','d'],
         ['00','31195','d'],
         ['1009','31195','d'],
         ['1001','31195','d'],
         ['1002','31195','d'],
         ['1003','31195','d'],
         ['1016','31195','d'],
         ['1004','31195','d'],
         ['1005','31195','d'],
         ['1007','31195','d']  
         ]
shuju_df = bm.BumenData().getdata('20190101','20191009',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)

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
test['公司去年同期取供比'] = test['广州自来水公司']['取供比'] - test['广州自来水公司']['取供比'].diff(365) 
test['西江原水管理所去年同期取供比'] = test['西江原水管理所']['取供比'] - test['西江原水管理所']['取供比'].diff(365) 
test['西村水厂去年同期取供比'] = test['西村水厂']['取供比'] - test['西村水厂']['取供比'].diff(365)
test['石门水厂去年同期取供比'] = test['石门水厂']['取供比'] - test['石门水厂']['取供比'].diff(365)
test['北部水厂去年同期取供比'] = test['北部水厂']['取供比'] - test['北部水厂']['取供比'].diff(365)
test['新塘水厂去年同期取供比'] = test['新塘水厂']['取供比'] - test['新塘水厂']['取供比'].diff(365)
test['西洲水厂去年同期取供比'] = test['西洲水厂']['取供比'] - test['西洲水厂']['取供比'].diff(365)
test['南洲水厂去年同期取供比'] = test['南洲水厂']['取供比'] - test['南洲水厂']['取供比'].diff(365)
test.sort_values('QUOTA_DATE')
test.to_excel(r'.\输出\20191009累计取供水量数据.xlsx')
