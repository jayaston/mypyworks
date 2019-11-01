# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:11:36 2019

@author: XieJie
"""

from texttable import Texttable

def print_table(df):
    
    #data=[{"name":"Amay","age":20,"result":80},
    #   {"name":"Tom","age":32,"result":90}]
    #df=pd.DataFrame(data,columns=['name','age','result'])
    
    tb=Texttable()
    #tb.set_cols_align(['l','r','r'])
    #tb.set_cols_dtype(['t','i','i'])
    #‘set_cols_align是对水平位置上的左中右靠齐。‘l'表示向左。‘c'表示居中,'r'向右。
    #['t', # text
    #'f', # float (decimal)
    #'e', # float (exponent)
    #'i', # integer
    #'a' # automatic]
    tb.header(df.columns.to_numpy())
    tb.add_rows(df.values,header=False)
    print(tb.draw())
    