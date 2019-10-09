# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:39:30 2019

@author: XieJie
"""
#sys.path.append(r'E:\pyworks\StatLedger\module')
#import sys
import pandas as pd
#import numpy as np
from tjfxdata import TjfxData
import re  
import os 
#import cx_Oracle
def order_gongshiku():
    
    dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(dir,"数据表/日数据计算公式库.xlsx"))    
    gongshiku = pd.read_excel(path,dtype='object')
    gongshiku = gongshiku.drop(['zbming'],axis=1)
    
    gongshiku['zhibiao'] = gongshiku['RECORD_TYPE']+'_'+gongshiku['var_dept']+'_'+gongshiku['var_code']
        
    
    def newformula(x):#对通用公式进行还原函数
        result = re.sub(r'(?=\b_)', x['var_dept'],x['setformula'])
        result = re.sub(r'(?<=_\b)', x['var_code'],result)
        result = re.sub(r'(?=\b\d+_)', x['RECORD_TYPE']+'_',result)
        return result
    
    gongshiku['setformula'] = gongshiku.apply(newformula,axis=1)
    gongshiku['zhibiaoji'] = gongshiku.apply(lambda x: re.findall(r'\b[a-z]_\d+_\d+\b',x['setformula']),axis=1)
    gongshiku.drop(['var_code','var_dept'],axis=1,inplace=True)      
    
    return gongshiku
gongshiku = order_gongshiku()
def get_castdata(startd,endd,quotalist): 
    thequotalist=[i.split('_') for i in quotalist]    
    tmp = []
    for i in thequotalist:
        i.append(i.pop(0))
        tmp.append(i)    
    quotalistall = tmp    
    result = TjfxData().getdata(startd,endd,quotalistall)
    result.QUOTA_VALUE = pd.to_numeric(result.QUOTA_VALUE,errors='coercs').fillna(0)
    result_dcast = pd.pivot_table(result,index='QUOTA_DATE',
                                columns=['RECORD_TYPE','QUOTA_DEPT_CODE','QUOTA_CODE'],
                                values = 'QUOTA_VALUE',fill_value=0 )  
    new_colnames = ["_".join(list(i)) for i in list(result_dcast.columns)]    
    result_dcast.columns=new_colnames
    return result_dcast



def inter_calcu(startd,endd,quotalist):
    zhibiaoji=[]
    for i in quotalist:
        zhibiaoji += gongshiku[gongshiku.zhibiao == i].iat[0,3]
    zhibiaoji = list(set(zhibiaoji))
    zhibiaojidict = all_calcu(startd,endd,zhibiaoji)
    for i in zhibiaojidict.keys():
        exec(i + "= zhibiaojidict.get(i,0)")        
    interquotadict = {}
    for i in quotalist:
        formula = gongshiku[gongshiku.zhibiao == i].iat[0,1]
        interquotadict.setdefault(i,eval(formula))        
    return interquotadict


def out_calcu(startd,endd,quotalist):
    diffquotadict = get_castdata(startd,endd,quotalist).to_dict('list')
    resultdict = {i:sum(diffquotadict.get(i,[0])) for i in quotalist}
    return resultdict

def all_calcu(startd,endd,quotalist):
    diffquotalist = list(set(quotalist) - set(gongshiku.zhibiao))
    interquotalist = list(set(quotalist) & set(gongshiku.zhibiao))
    if len(interquotalist) == 0: 
        resultdict = out_calcu(startd,endd,diffquotalist)
    else:
        if len(diffquotalist)==0:
            resultdict = inter_calcu(startd,endd,interquotalist)
        else:
            dic1 = out_calcu(startd,endd,diffquotalist)
            dic2 = inter_calcu(startd,endd,interquotalist)
            resultdict = dict(dic1,**dic2) 
    return resultdict
            




if __name__ == "__main__" :  
    
    startd,endd,quotalist = '20190901','20190926',['d_00_01464','d_1001_01464','d_1002_01464','d_1003_01464']
    test = all_calcu(startd,endd,quotalist)

