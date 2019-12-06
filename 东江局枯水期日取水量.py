#!/usr/bin/env python
# coding: utf-8

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
shuju_df = tjfx.TjfxData().getdata('20191001','20191130',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


shuju_df = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=['QUOTA_NAME'],values= 'QUOTA_VALUE',aggfunc='sum')

shuju_df.columns=['东江取水量']
shuju_df.to_excel(r".\mypyworks\输出\2019年10-11月东江取水量.xlsx")



