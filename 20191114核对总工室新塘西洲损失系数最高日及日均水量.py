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
list1 = [['1004','04281','m'],
         ['1005','04281','m'],
         ['1004','00752','m'],
         ['1005','00752','m']
         ]
shuju_df = tjfx.TjfxData().getdata('20140101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)



print(shuju_df)


shuju_df1 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
                          aggfunc=np.sum)

shuju_df1=shuju_df1.resample("Y").sum().applymap(lambda x :round(x/10000,2))
#新塘西洲率输水及制水合计损失系数
shuju_df1.eval("损失系数=(取水量-水厂供水量)/取水量*100",inplace=True)
shuju_df1['损失系数'].mean()


list2 = [['1004','04281','d'],
         ['1005','04281','d'],
         ['1004','00752','d'],
         ['1005','00752','d']
         ]
shuju_df = tjfx.TjfxData().getdata('20140101','20181231',list2)
shuju_df.info()
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

shuju_df2 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
                          aggfunc=np.sum)
shuju_df2.reset_index(inplace=True)
shuju_df2.info()
#按年最大值
get_max = lambda x: x[x['取水量']==x['取水量'].max()][['QUOTA_DATE','取水量']]
                        
# python就是灵活啊。
get_max.__name__ = "maxday"
shuju_df2.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y')]).apply(get_max)
shuju_df2.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y')]).mean()


