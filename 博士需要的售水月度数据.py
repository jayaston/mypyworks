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


list1 = [['00','00718','m'], #供水总量 
          ['1001','00718','m'], #西村供水总量 
          ['1002','00718','m'], #石门供水总量 
          ['100301','00718','m'], #江一供水总量 
          ['100302','00718','m'], #江二供水总量 
          ['1004','00718','m'], #新塘供水总量 
          ['1005','00718','m'], #西洲供水总量 
          ['1007','00718','m'], #南洲供水总量 
          ['1016','00718','m'], #北部供水总量   
         ['00','00409','m'],#售水量
         ['0900','30984','m'],#东区供水量
          ['0900','00409','m'],#东区售水量
          ['0901','30984','m'],#中区供水量
          ['0901','00409','m'],#中区售水量
          ['0902','30984','m'],#南区供水量
          ['0902','00409','m'],#南区售水量
          ['0903','30984','m'],#北区供水量
          ['0903','00409','m'],#北区售水量
         ['00','23425','m'],#发单水量
         ['00','23594','m'],#两月发单
         ['00','23765','m'],#每月发单
         ['00','00311','m'],#产值   
         ['00','31594','m'],#抄表到户水量
         ['00','11930','m'],#对外取水量
         ['00','04281','m'],#取水量
         ['00','29891','m'],#销售收入
         ['00','37971','m'],#二阶梯收入
         ['00','34097','m']#三阶梯收入
         ]


list2 = [['00','00718','d'] #日供水总量 
         ]

shuju_df = tjfx.TjfxData().getdata('19900101','20200731',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
# test.columns
# test = test.reindex(columns=[
#             ('广州自来水公司',    '净水售水量'),
#             ('广州自来水公司',   '水厂供水总量'),
#             ('广州自来水公司',   '发单水量合计'),
#             ('广州自来水公司', '两月发单水量合计'),
#             ('广州自来水公司', '每月发单水量合计')
#             ])
# test.info()

##按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon<='10']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
#test = test.resample("Y").sum()

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\1990-2020年指标预测月年数据.xlsx')
