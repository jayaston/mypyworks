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
import shujuyuan as sj
#import re    



list1 = [['1002','04281','m'],
         ['1002','00718','m'],
         ['1003','04281','m'],
         ['1003','00718','m']
        ]
shuju_df = sj.Datataizhang().getdata('20130101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)

#shuju_df.to_excel("E:\\pyworks\\2019年供水.xls")
#按年汇总
shuju_leiji = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_DEPT_CODE','QUOTA_CODE','QUOTA_NAME','GROUP_NAME']).sum()
shuju_leiji.to_excel("E:\\pyworks\\累计数据.xls")
#按年平均
shuju_mean = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_DEPT_CODE','QUOTA_CODE','QUOTA_NAME','GROUP_NAME']).mean()
#shuju_mean.to_excel("E:\\pyworks\\2019周报平均日供水总量.xls")

#按年最大值
get_max = lambda x: x[x.QUOTA_VALUE==x.QUOTA_VALUE.max()][['QUOTA_DATE','QUOTA_VALUE']]
# python就是灵活啊。
get_max.__name__ = "max day"



shuju_max = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_NAME','GROUP_NAME']).apply(get_max)

shuju_gszl = shuju_df[(shuju_df['QUOTA_DEPT_CODE']=='00') & (shuju_df['QUOTA_CODE']=='00718')]
shuju_maxgszl = shuju_gszl[shuju_gszl.QUOTA_VALUE == shuju_gszl.QUOTA_VALUE.max()]

shuju_gszl.QUOTA_VALUE.mean()

print(shuju_leiji)