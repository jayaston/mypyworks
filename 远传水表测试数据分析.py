# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:10:53 2021

@author: Jay
"""
import pandas as pd
import numpy as np
import datetime as dt

startd='2021-8-21'
endd = '2021-10-13'

meterinfo = pd.read_excel('d:\\专题工作（重要）\\远传表\\测试清单.xlsx',dtype={'客户编号':str,'智能表码':str})
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


testdate = pd.date_range(start=startd, end=endd,  closed=None,)
testdate = pd.Series(testdate).dt.strftime('%m-%d').str.lstrip("0").str.replace("-0", "-")
meterdata  = pd.DataFrame()
for i in testdate:    
    df = pd.read_excel('d:\\专题工作（重要）\\远传表\原始数据汇总\测试表'+i+'原始数据.xlsx',usecols=['站号','时间','P20'],
                       )
    df = df[df['站号']!='站号']
    meterdata = pd.concat([meterdata,df],ignore_index=1)

meterdata.columns=['表码','时间','行度']
meterdata['日期'] = pd.to_datetime(meterdata['时间']).dt.strftime('%Y-%m-%d')
meterdata['年月'] = pd.to_datetime(meterdata['时间']).dt.strftime('%Y%m')

meterdata.dropna(subset=['表码','时间'],inplace=True)

list_wrong_meter1 = list(meterinfo[meterinfo['厂家']=='迈拓']['智能表码'].unique())
list_wrong_date1 = ['2021-09-04','2021-09-05','2021-09-06']

list_wrong_meter2 = list(meterinfo[meterinfo['厂家']=='威铭']['智能表码'].unique())
list_wrong_date2 = ['2021-09-17']

list_wrong_meter3 = list(meterinfo[meterinfo['厂家']=='东海']['智能表码'].unique())
list_wrong_date3 = list(pd.date_range('2021-09-28','2021-10-7').strftime('%Y-%m-%d'))

meterdata = meterdata[~(
                        ( meterdata['表码'].isin(list_wrong_meter1) & meterdata['日期'].isin(list_wrong_date1) ) |
                        ( meterdata['表码'].isin(list_wrong_meter2) & meterdata['日期'].isin(list_wrong_date2) ) |
                        ( meterdata['表码'].isin(list_wrong_meter3) & meterdata['日期'].isin(list_wrong_date3) ) 
                        )]

meterdf1 = meterdata.groupby(['表码','日期']).apply(lambda df:df['行度'].count()/len(df['行度']))
meterdf1 = meterdf1.reset_index()
meterdf1.columns =['表码','日期','抄见率']
meterdf1 = meterdf1.groupby(['表码']).apply(lambda df:len(df[df['抄见率']>0])/len(df['抄见率']))
meterdf1 = pd.DataFrame(meterdf1.values.T,columns=['上线天数比'],index=meterdf1.index).reset_index().query('上线天数比<1').sort_values('上线天数比')

meterdf = meterdata.groupby(['表码','日期']).apply(lambda df:df['行度'].count()/len(df['行度']))
meterdf = meterdf.reset_index()
meterdf.columns =['表码','日期','抄见率']
meterdf = meterdf.groupby(['表码']).apply(lambda df:len(df[df['抄见率']==1])/len(df['抄见率']))
meterdf = pd.DataFrame(meterdf.values.T,columns=['全抄天数比'],index=meterdf.index).reset_index().query('全抄天数比<1')

dfmerge = pd.merge(meterdata,meterinfo,how='left',left_on='表码', right_on='智能表码')
dfmerge = dfmerge.groupby(['厂家','表码']).apply(lambda df:df['行度'].count()/len(df['行度'])).reset_index()
dfmerge.columns =['厂家','表码','抄见率']

wuchadf1= pd.read_excel('d:\\专题工作（重要）\\远传表\\手抄数及相对偏差汇总.xls',sheet_name='0910', usecols=['智能表码','手抄行度','系统行度'],dtype={'智能表码':str})
wuchadf1['日期']='2021-09-10'
wuchadf2= pd.read_excel('d:\\专题工作（重要）\\远传表\\手抄数及相对偏差汇总.xls',sheet_name='0918', usecols=['智能表码','手抄行度','系统行度'],dtype={'智能表码':str})
wuchadf2['日期']='2021-09-18'
wuchadf3= pd.read_excel('d:\\专题工作（重要）\\远传表\\手抄数及相对偏差汇总.xls',sheet_name='1015', usecols=['智能表码','手抄行度','系统行度'],dtype={'智能表码':str})
wuchadf3['日期']='2021-10-15'

wuchadf = pd.concat([wuchadf1,wuchadf2,wuchadf3],ignore_index=1)
wuchadf['绝对误差'] = abs(wuchadf['系统行度']-wuchadf['手抄行度']) 
wuchadf['相对偏差率'] = wuchadf['绝对误差']/wuchadf['手抄行度']








dfmerge['抄见率']= dfmerge['抄见率'].round(4)

import seaborn as sns
#sns.set_theme(style="whitegrid")
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
from matplotlib import ticker

bins=[0.85,0.9,0.95,0.98,1]

xticks=[0.875,0.925,0.965,0.99]
xlabel=['85~90%','90~95%','95~98%','98%以上']

fig=plt.figure(figsize=(6,4),dpi=100)
#fig.suptitle("我是画布的标题",fontsize=20)
g=sns.histplot(data=dfmerge, x="抄见率",kde=True,hue="厂家",bins=bins, multiple="dodge")
sns.despine() #移除边框
#plt.xticks([-16,-6,4,14,24,34,44],labels=['-10以下','-10~0','0~10','10~20','20~30','30~40','40以上'])
plt.ylabel('水表数量分布')
plt.xlabel('抄见率')
plt.xticks(ticks=xticks,labels=xlabel)




