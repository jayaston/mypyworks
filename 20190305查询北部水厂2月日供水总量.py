# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""

import sys
sys.path.append('E:\\pyworks')
import pandas as pd
import numpy as np
import ShuJuCaoZuo.shujuyuan as sj
#import re    



list1 = [['1016','00718','d']]
shuju_df = sj.Datataizhang().getdata('20190201','20190228',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)


shuju_df.to_excel("E:\\pyworks\\2月份北部水厂日供水量.xls")
