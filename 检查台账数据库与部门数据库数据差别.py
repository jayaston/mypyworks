# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:21:48 2019

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
import bumendata as bm
a = tjfx.TjfxData()
df_a = a.getdata(startd='20190221',endd='20190222')
b = bm.BumenData()
df_b = b.getdata('20190131','20190221')

df = pd.merge(df_a,df_b,how="left",on=['QUOTA_DATE','QUOTA_DEPT_CODE','QUOTA_CODE','RECORD_TYPE'])

df[['QUOTA_VALUE_x','QUOTA_VALUE_y']]=df[['QUOTA_VALUE_x','QUOTA_VALUE_y']].astype(dtype='float',errors='ignore')

df2 = df.assign(wucha=lambda x:pd.to_numeric(x.QUOTA_VALUE_x,errors='coerce').fillna(0)\
                -pd.to_numeric(x.QUOTA_VALUE_y,errors='coerce').fillna(0))
df2['RECORD_TYPE'] = pd.to_str(df2['RECORD_TYPE'])
df2.info()


df3 = df2[(df2.wucha>0) & (df2.RECORD_TYPE  == 'd')]



