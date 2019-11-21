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



list1 = [['1016','00718','d'],
         
         
        ]
shuju_df = tjfx.TjfxData().getdata('20190112','20191118',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

shuju_df['mon'] = shuju_df.QUOTA_DATE.dt.strftime('%m')
shuju_df['d'] = shuju_df.QUOTA_DATE.dt.strftime('%d')

test = pd.pivot_table(shuju_df,index = ['d'],columns = ['mon'],values='QUOTA_VALUE')

try:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),r"输出\2019北部水厂供水量.xlsx"))
    test.to_excel(path)    
except:
    test.to_excel(r'./mypyworks/输出/2019北部水厂供水量.xlsx')


