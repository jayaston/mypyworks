# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:30:43 2019

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
list1 = [['1004','04281','d'],
         ['1005','04281','d']
         
         ]
shuju_df = tjfx.TjfxData().getdata('20170401','20200331',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)



print(shuju_df)


shuju_df1 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
                          aggfunc=[np.sum])


shuju_df1.reset_index(inplace=True)
shuju_df1.columns=['日期','刘屋洲取水量']

import datetime as dt
shuju_df1['年月']=shuju_df1['日期'].dt.strftime('%Y%m')
shuju_df1['日']=shuju_df1['日期'].dt.strftime('%d')+'日'

shuju_df2 = pd.pivot_table(shuju_df1,index=['年月'],columns=[ '日'],values= '刘屋洲取水量')

shuju_df2.to_excel(r'./mypyworks/输出/刘屋洲取水日台账.xls')




