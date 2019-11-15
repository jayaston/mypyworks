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
def order_gongshiku(sheet_name="d"):
    
    dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(dir,"数据表/全新公式库北部太和.xlsx"))    
    gongshiku = pd.read_excel(path,sheet_name=sheet_name,dtype={'var_code':str,'var_dept':str})
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


def get_castdata(startd,endd,quotalist): 
    thequotalist=[i.split('_') for i in quotalist]    
    tmp = []
    for i in thequotalist:
        i.append(i.pop(0))
        tmp.append(i)    
    quotalistall = tmp    
    result = TjfxData().getdata(startd,endd,quotalistall)
    result.QUOTA_VALUE = pd.to_numeric(result.QUOTA_VALUE,errors='coerce').fillna(0)
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
    diffquotadf = get_castdata(startd,endd,quotalist).reindex(columns=quotalist,fill_value=0)
    diffquotadict = diffquotadf.to_dict('series') 
#    resultdict = {i:diffquotadict.get(i,0) for i in quotalist}

    return diffquotadict

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
            
def calcu_tjfx(startd,endd,typer="d"):
    gongshiku = order_gongshiku(typer)
    quotalist = list(set(gongshiku['zhibiao']))
    resultdict = all_calcu(startd,endd,quotalist)
    resultlist= []
    for i in resultdict.keys():
        for j,k in list(zip(resultdict[i].index,resultdict[i])):            
            value = (i.split('_')[2],j.strftime('%Y%m'),j.strftime('%Y-%m-%d %H:%M:%S'),'%.2f'%float(k),'',i.split('_')[1],'','',i.split('_')[0])
            resultlist.append(value)
    TjfxData().importdata(resultlist)
    
   
#小时累计计算到日，与台账已有日数据比较。日累计计算月，与台账已有月数据比对

if __name__ == "__main__" :    
    startd,endd = '20191001','20191028'
    gongshiku = order_gongshiku("h")
    quotalist = list(set(gongshiku['zhibiao']))
    resultdict = all_calcu(startd,endd,quotalist)
    resultlist= []
    for i in resultdict.keys():
        for j,k in list(zip(resultdict[i].index,resultdict[i])):            
            value = (i.split('_')[2],j.strftime('%Y%m'),j.strftime('%Y-%m-%d %H:%M:%S'),'%.2f'%float(k),'',i.split('_')[1],'','',i.split('_')[0])
            resultlist.append(value)
    

