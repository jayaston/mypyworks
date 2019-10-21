# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:25:43 2019

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
import re   

dongjianglist = [['1004','04281','d'],['1005','04281','d']]

xintangxizhou = tjfx.TjfxData().getdata('20140101','20181231',dongjianglist)
xintangxizhou['QUOTA_VALUE'] = pd.to_numeric(xintangxizhou['QUOTA_VALUE'],errors='coercs').fillna(0)
dongjiang = xintangxizhou.set_index('QUOTA_DATE').resample('d')['QUOTA_VALUE'].sum()
dongjiang = pd.DataFrame(dongjiang)
dongjiang.info()
get_max = lambda x: x[x['QUOTA_VALUE']==x['QUOTA_VALUE'].max()]
# python就是灵活啊。
get_max.__name__ = "maxd"


dongjiang.resample('Y').agg(get_max)