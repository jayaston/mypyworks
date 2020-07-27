# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:17:16 2020

@author: XieJie
"""

import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')

from bumendata import BumenData
from tjfxdata import TjfxData
import pandas as pd
import numpy as np
import os
import datetime as dt
import re
#os.getcwd()
#同步数据表


#同步指标信息表
#提取指标表
# def fresh_Quota_Define():  
df_quota = TjfxData().get_all_quota()
#写入指标表
BumenData().imp_tbl(df_quota,'CS_QUOTA_DEFINE')

#同步部门信息表
#提取部门表
df_dept = TjfxData().get_all_dept()
#写入部门表
BumenData().imp_tbl(df_dept,'HR_ORGANIZATION') 


#同步公式信息表
#提取系统公式表
formulaset = TjfxData().get_formula().query("TZ_TYPE != 'd' ")#提取公式概要表
formulaset = formulaset.rename(columns = {'QUOTA_NAME':'zbming','QUOTA_CODE':'var_code','ZB_DEPT_CODE':'var_dept','TZ_TYPE':'RECORD_TYPE'})
formulaset = formulaset.reindex(columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','END_TIME','FORMULA_CODE','FORMULA','方案','目录'])
#len(formulaset['FORMULA_CODE'].unique())
#提取公式详情表
formuladetail=TjfxData().get_formula_detail() 
formuladetail = formuladetail[formuladetail['FORMULA_CODE'].isin( set(formulaset['FORMULA_CODE']))]
formuladetail.sort_values(['FORMULA_CODE','FORMULA_ORDER_SN','FLOW_NO'],inplace=True)#增加排序字段，第115行选择公式的范围也需要修改
formuladetail = formuladetail.fillna(value='')#填充空值
formuladetail['PARAMETER']=formuladetail['PARAMETER'].astype('str')#参数列改成字符型
#改变计算符号
formuladetail.replace({'OPERATION':{"0":"+","1":"-","2":"*","3":"/","4":"",
                          "5":"MaxHNum","6":"MaxH","7":"MinHNum","8":"MinH",
                          "9":"MaxDNum","10" :"MaxD","11":"MinDNum","12":"MinD",
                          "13":"前n期"}},inplace=True)
#插入空格和-
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_CODE'),'lianjie')
formuladetail=formuladetail.reindex(columns=col_name,fill_value='_')
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_DEPT_CODE'),'space1')
col_name.insert(col_name.index('RIGHT_BRACKET'),'space2')
formuladetail=formuladetail.reindex(columns=col_name,fill_value=' ')


#移除汉和特殊字符
def removeChnAndCharacter(str1):
    #将中文标点符号转换为英文标点符号
    def C_trans_to_E(string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        #ord返回ASCII码对应的int
        #zip将合并为列表，元素为元祖，元祖为对应位置所有元素依次的集合，如这种形式[(',','，')...]
        #s生成对应字典
        table= {ord(f):ord(t) for f,t in zip(C_pun,E_pun)}
        #将字符传对应转换
        return string.translate(table)
    C_pun = u'，。！？【】（）《》“‘'
    strTmp = ''
    if not isinstance(str1,str):
        return strTmp
    for i in range(len(str1)):
        #中文字符范围
        #https://blog.csdn.net/qq_22520587/article/details/62454354
        if str1[i] >= u'\u4e00' and str1[i] <= u'\u9fa5' \
                or str1[i] >= u'\u3300' and str1[i] <= u'\u33FF' \
                or str1[i] >= u'\u3200' and str1[i] <= u'\u32FF' \
                or str1[i] >= u'\u2700' and str1[i] <= u'\u27BF' \
                or str1[i] >= u'\u2600' and str1[i] <= u'\u26FF' \
                or str1[i] >= u'\uFE10' and str1[i] <= u'\uFE1F' \
                or str1[i] >= u'\u2E80' and str1[i] <= u'\u2EFF' \
                or str1[i] >= u'\u3000' and str1[i] <= u'\u303F' \
                or str1[i] >= u'\u31C0' and str1[i] <= u'\u31EF' \
                or str1[i] >= u'\u2FF0' and str1[i] <= u'\u2FFF' \
                or str1[i] >= u'\u3100' and str1[i] <= u'\u312F' \
                or str1[i] >= u'\u21A0' and str1[i] <= u'\u31BF' \
                :
            pass#中文字符不处理
        else:
            if str1[i] in C_pun:
                # st = C_trans_to_E(str1[i])#中文标点不处理
                pass
            else:
                st = str1[i]
            strTmp += st
    return strTmp

formuladetail['QUOTA_DEPT_CODE']= formuladetail['QUOTA_DEPT_CODE'].apply(removeChnAndCharacter)
#构造公式表达式函数按
def get_formula(df):
    nrows=len(df)
    formula_exp = ''
    for i in range(nrows):
        for j in range(2,10) :            
            formula_exp += df.iat[i,j]
    return formula_exp 
        
formuladetail = formuladetail.groupby(['FORMULA_CODE']).apply(get_formula)
formuladetail = formuladetail.str.replace(' _ ','')
formuladetail=pd.DataFrame(formuladetail)
formuladetail.columns=['setformula']                
formuladetail.reset_index(inplace=True)  

#系统公式表
formula_tjfx = pd.merge(formulaset,formuladetail,how='left',on=['FORMULA_CODE'])
formula_tjfx = formula_tjfx.reindex(columns=['zbming', 'var_code', 'var_dept', 'RECORD_TYPE', 'START_TIME',
       'END_TIME', 'FORMULA_CODE', 'setformula','FORMULA', '方案', '目录'])

#提取本地公式
# dir = os.path.dirname(os.path.dirname(__file__))
# path = os.path.abspath(os.path.join(dir,"数据表/全新公式库北部太和.xlsx"))    
formula_excel = pd.read_excel(r'./mypyworks/StatLedger/数据表/新格式公式表.xlsx',
                          sheet_name='formula',dtype={'var_code':str,'var_dept':str})
#系统公式表与本地公式表合并
formula = pd.concat([formula_excel,formula_tjfx],ignore_index=True)
formula.info()
#写入mysql数据库
BumenData().imp_tbl(formula,'FORMULA') 

#读取公式表
formula = BumenData().get_formula()
#提取公式的开始和结束时间
formula_time = sorted(set(pd.concat((formula['START_TIME'],formula['END_TIME']),axis=0)))
formula_time.remove(pd.NaT)#删除非时间类型
#增加时期列表的端点
formula_time[0:0] = [dt.datetime(1900,1,1)]
formula_time[len(formula_time):len(formula_time)] = [dt.datetime(2099,12,31)]
#将公式表按时间段分开
time_tuple = [(formula_time[i],formula_time[i+1]) for i in range(len(formula_time)-1)]#建立时间范围元组的列表
#构建key为元组的字典。
formula_dict={}
for i in range(len(formula_time)-1):
    formula_dict[time_tuple[i]]= formula[(formula['START_TIME']<=formula_time[i]) & (formula['END_TIME'] >= formula_time[i+1])]

#每个时间段公式库的的冲突公式
formula_conflict = pd.DataFrame()
for i in formula_dict.keys():
    df = formula_dict[i]
    df = df[df.duplicated(['var_code','var_dept','RECORD_TYPE'],keep= False)]\
    .sort_values(['var_code','var_dept','RECORD_TYPE'])
    formula_conflict = pd.concat([formula_conflict,df])
formula_conflict.drop_duplicates(inplace=True)
n_formula_conflict = formula_conflict.groupby(['var_code','var_dept','RECORD_TYPE']).agg({'FORMULA_CODE': pd.Series.nunique})#查看冲突指标数量
formula_conflict.to_excel(r'./mypyworks/StatLedger/数据表/冲突公式表.xlsx')

#冲突公式强制选取第一条公式(慎重使用)
for i in formula_dict.keys():
    df = formula_dict[i]
    df = df.drop_duplicates(['var_code','var_dept','RECORD_TYPE'],keep='first')
    formula_dict[i] =df


#运算和结束时间    
startd=dt.datetime.strptime('20200401','%Y%m%d')
endd=dt.datetime.strptime('20200725','%Y%m%d')
#加入formulatime排序，用推导式求范围内的个数
formula_time.extend([startd,endd])
formula_time.sort()
node_list = [x for x in formula_time if x>=startd and x<endd]
#接下来分三种情况讨论，第一种》=start、《end 之间没有日期元素，计算日期为satrt end，计算库为start-1，end+1，
def order_gongshiku(gongshiku): 
        gongshiku['zhibiao'] = gongshiku['RECORD_TYPE']+'_'+gongshiku['var_dept']+'_'+gongshiku['var_code']
        gongshiku['setformula'] = gongshiku['setformula'].astype('str')
        def newformula(x):#对通用公式进行还原函数
            result = re.sub(r'(?=\b_)', x['var_dept'],x['setformula'])
            result = re.sub(r'(?<=_\b)', x['var_code'],result)
            result = re.sub(r'(?=\b\d+_)', x['RECORD_TYPE']+'_',result)
            return result
        
        gongshiku['setformula'] = gongshiku.apply(newformula,axis=1)
        gongshiku['zhibiaoji'] = gongshiku.apply(lambda x: re.findall(r'\b[a-z]_\d+_\d+\b',x['setformula']),axis=1)
        gongshiku.drop(['var_code','var_dept'],axis=1,inplace=True)
        return gongshiku
if len(node_list) == 0:
    gongshiku = formula_dict[formula_time[formula_time.index(startd)-1],formula_time[formula_time.index(endd)+1]]
    gongshiku = gongshiku.drop(['zbming','START_TIME','END_TIME'],axis=1)
    gongshiku = order_gongshiku(gongshiku)
    
'FORMULA_CODE','FORMULA','方案','目录',
#第二种情况有一个，start到这一个，用前一个到这个公式库，这一个到end的数据，用这一个到后一个公式库。
#第三种情况，有两个及以上，两个中间完整区间的数据和公式库，两端start到后一个最近的时间，应用start前一个到后一个的公式库，end也是如此。






#验证是否还有同一个指标
test2[test2.duplicated(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE'],keep= False)]
test2.info()
test2 = test2.rename(columns = {'QUOTA_NAME':'zbming','QUOTA_CODE':'var_code','ZB_DEPT_CODE':'var_dept','TZ_TYPE':'RECORD_TYPE'})
test2 = test2.reindex(columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','START_TIME','FORMULA_CODE'])













#提取时间指标
df = TjfxData().getdata('20200701','20200720')
df_h = df.query("RECORD_TYPE=='h'")

#时指标公式校验并提示错误

#时指标写入数据表

#向上聚合成日指标


#与日指标合并提示差异

#日指标公式校验并提示错误数据

#日指标写入数据库

#向上聚合成月指标



#和月指标合并并提示差异

#月指标公式校验并提示错误

#月指标写入数据库

#向上汇聚成年指标

#写如mysql数据库



    
    
        
    
    

    