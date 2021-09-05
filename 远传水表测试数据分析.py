# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:10:53 2021

@author: Jay
"""
import pandas as pd
import numpy as np
import datetime as dt

meterinfo = pd.read_excel('d:\\工作\\远传表\\测试清单.xlsx',dtype={'客户编号':str,'智能表码':str})
meterinfo['智能表生产厂家'].unique()
meterinfo['厂家'] = meterinfo['智能表生产厂家'].str.replace('唐山汇中仪表股份有限公司','汇中')\
    .replace('江苏迈拓智能仪表有限公司','迈拓')\
        .replace('宁波水表（集团）股份有限公司','宁波')\
            .replace('金卡水务科技有限公司','金卡')\
                .replace('湖南威铭能源科技有限公司','威铭')\
                    .replace('宁波东海集团有限公司','东海')\
                        .replace('汇中仪表股份有限公司','汇中')\
                            .replace('三川智慧科技股份有限公司','三川')
meterinfo['水表类型'] = meterinfo['厂家'].apply(lambda x:'超声波' if x in ['迈拓','汇中'] else '机械表')                        
                    

testdate = pd.date_range(start='2021-8-21', end='2021-9-1',  closed=None,)
testdate = pd.Series(testdate).dt.strftime('%m-%d').str.lstrip("0").str.replace("-0", "-")
meterdata  = pd.DataFrame()
for i in testdate:    
    df = pd.read_excel('d:\工作\远传表\原始数据汇总\测试表'+i+'原始数据.xlsx',usecols="A:E",
                       names=['楼栋','详细地址','表码','时间','行度'])
    df = df[df['表码']!='站号']
    meterdata = pd.concat([meterdata,df],ignore_index=1)

meterdata['日期'] = pd.to_datetime(meterdata['时间']).dt.strftime('%Y-%m-%d')
meterdata['年月'] = pd.to_datetime(meterdata['时间']).dt.strftime('%Y%m')

meterdf = meterdata.groupby(['表码','日期']).apply(lambda df:df['行度'].count()/len(df['行度']))
meterdf = meterdf.reset_index()
meterdf.columns =['表码','日期','抄见率']
meterdf = meterdf.groupby(['表码']).apply(lambda df:len(df[df['抄见率']==1])/len(df['抄见率']))
meterdf = pd.DataFrame(meterdf.values.T,columns=['全抄天数比'],index=meterdf.index).reset_index().query('全抄天数比<1')
