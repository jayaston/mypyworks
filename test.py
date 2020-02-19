# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:44:39 2020

@author: Jay
"""
import numpy as np
import pandas as pd
dic1 = {'a':10,'b':20,'c':30,'d':40,'e':50}
dic1
type(dic1)
s2 = pd.Series(dic1)
s2
type(s2)

arr2 = np.array(np.arange(12)).reshape(4,3)
arr2 = arr2.tolist()
type(arr2)
df1 = pd.DataFrame(arr2)
df1
type(df1)

np.random.randn(4,4)


dict = {'Google': ['www.google.com','a'], 'Runoob': ['www.runoob.com','a'], 'taobao': ['www.taobao.com','a']}
 
list(dict.items())
df2 = pd.DataFrame.from_items(list(dict.items()))
# 遍历字典列表
for key,values in  dict.items():
    print key,values
print(df2['Google'].unique)

df2[df2=='a']='B'