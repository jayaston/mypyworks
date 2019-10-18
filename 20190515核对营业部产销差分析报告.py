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
    sys.path.append(r'.\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    



list1 = [['00','00718','d']
         ]
shuju_df = tjfx.TjfxData().getdata('20180101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)

#shuju_df.to_excel("E:\\pyworks\\2019年供水.xls")
shuju_leiji = shuju_df.groupby(['QUOTA_DEPT_CODE','QUOTA_CODE']).sum()
shuju_leiji['QUOTA_VALUE'] = shuju_leiji['QUOTA_VALUE'].apply(lambda x: '%.2f'%(x/10000))
#shuju_leiji.to_excel("E:\\pyworks\\2019周报累计供水总量.xls")
shuju_mean = shuju_df.groupby(['QUOTA_DEPT_CODE','QUOTA_CODE']).mean()
shuju_mean['QUOTA_VALUE'] = shuju_mean['QUOTA_VALUE'].apply(lambda x: '%.2f'%(x/10000))
#shuju_mean.to_excel("E:\\pyworks\\2019周报平均日供水总量.xls")
shuju_gszl = shuju_df[(shuju_df['QUOTA_DEPT_CODE']=='00') & (shuju_df['QUOTA_CODE']=='00718')]
shuju_maxgszl = shuju_gszl[shuju_gszl.QUOTA_VALUE == shuju_gszl.QUOTA_VALUE.max()]

shuju_gszl.QUOTA_VALUE.mean()

print(shuju_leiji)