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
list1 = [['1004','04281','m'],
         ['1005','04281','m'],
         ['1004','00718','m'],
         ['1005','00718','m']
         ]
shuju_df = tjfx.TjfxData().getdata('20140101','20190831',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)



print(shuju_df)


shuju_df = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=['GROUP_NAME', 'QUOTA_NAME'],values= 'QUOTA_VALUE')

shuju_df.to_excel(r"E:\pyworks\2014-2019年新塘西洲取供水量.xlsx")

dongjianglist = [['1004','00718','d'],['1005','00718','d']]
xintangxizhou = sj.Datataizhang().getdata('20140101','20190331',dongjianglist)
xintangxizhou['QUOTA_VALUE'] = pd.to_numeric(xintangxizhou['QUOTA_VALUE'],errors='coercs').fillna(0)
dongjiang = xintangxizhou.set_index('QUOTA_DATE').resample('d')['QUOTA_VALUE'].sum()
dongjiang = pd.DataFrame(dongjiang)
dongjiang.info()

get_max = lambda x: x[x['QUOTA_VALUE']==x['QUOTA_VALUE'].max()]

get_max.__name__ = "maxd"

dongjiang_gongshuiliang_max = dongjiang.resample('Y').agg(get_max)




dongjiang_gongshuiliang_max.to_excel(r"E:\pyworks\2014-2019年新塘西洲年最大日供水总量.xlsx")

