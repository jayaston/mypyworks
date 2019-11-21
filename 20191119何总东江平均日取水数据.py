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



list1 = [['1004','04281','d'],
         
         ['1005','04281','d']
         
        ]
shuju_df = tjfx.TjfxData().getdata('20180101','20181114',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

shuju_df1 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
                          aggfunc=np.sum)
shuju_df1.mean()






shuju_df1=shuju_df1.resample("Y").sum().applymap(lambda x :round(x/10000,2))






#shuju_df.to_excel("E:\\pyworks\\2019年供水.xls")
#按年汇总
shuju_leiji = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_DEPT_CODE','QUOTA_CODE','QUOTA_NAME','GROUP_NAME']).agg(['sum','mean'])
shuju_leiji.to_excel("C:\\Users\\XieJie\\mypyworks\\输出\\20191024提供总工室西江下陈取水量日最大日平均数据.xls")
#按年平均
#shuju_mean = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_DEPT_CODE','QUOTA_CODE','QUOTA_NAME','GROUP_NAME']).mean()
#shuju_mean.to_excel("E:\\pyworks\\2019周报平均日供水总量.xls")

#按年最大值
get_max = lambda x: x[x.QUOTA_VALUE==x.QUOTA_VALUE.max()][['QUOTA_DATE','QUOTA_VALUE']]
# python就是灵活啊。
get_max.__name__ = "max day"



shuju_max = shuju_df.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y'),'QUOTA_NAME','GROUP_NAME']).apply(get_max)
shuju_max.to_excel("C:\\Users\\XieJie\\mypyworks\\输出\\20191024提供总工室西江下陈取水量日最大日平均数据1.xls")

shuju_gszl = shuju_df[(shuju_df['QUOTA_DEPT_CODE']=='00') & (shuju_df['QUOTA_CODE']=='00718')]
shuju_maxgszl = shuju_gszl[shuju_gszl.QUOTA_VALUE == shuju_gszl.QUOTA_VALUE.max()]

shuju_gszl.QUOTA_VALUE.mean()

print(shuju_leiji)