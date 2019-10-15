# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:51:44 2019

@author: Jay
"""

import pandas as pd
che_xi_biao = pd.read_excel(r'.\汽车数据\车系表.xlsx')
che_xi_biao = che_xi_biao.drop(['车系.创建时间','车系.更新时间','车系.平台代码','车系.月份'],axis=1)
che_xi_biao.columns = ['车系名称', '车系状态', '品牌名称', '品牌ID', '汽车平台', '子品牌名称','子品牌ID', '车系ID']
che_xi_biao = che_xi_biao[['车系ID','车系名称','车系状态','子品牌ID','子品牌名称','品牌ID','品牌名称','汽车平台']]

che_xing_biao = pd.read_excel(r'.\汽车数据\车型表.xlsx')
che_xing_biao.info()
che_xing_biao = che_xing_biao.drop(['车型.车系年款ID','车型.创建时间','车型.更新时间','车型.平台代码','车型.月份'],axis=1)
che_xing_biao.columns = ['车系ID', '车系车型合并名称', '车型名称', '车型状态', '车型ID-PK', '汽车平台']


che_xing_biao = pd.merge(che_xing_biao,che_xi_biao,how='left',on='车系ID',validate='m:1')
che_xing_biao = che_xing_biao[['车型ID-PK','车型名称','车系车型合并名称','车型状态','车系ID','车系名称','车系状态',
                               '子品牌ID','子品牌名称','品牌ID', '品牌名称', '汽车平台_y']]
che_xing_biao.rename(columns={'车型ID-PK':'车型ID','汽车平台_y':'汽车平台'},inplace=True)







wen_zhang_nei_rong = pd.read_excel(r'.\汽车数据\文章内容.xlsx')
wen_zhang_xing_xi = pd.read_excel(r'.\汽车数据\文章信息.xlsx')

zhu_ti_xing_xi = pd.read_excel(r'.\汽车数据\主题信息.xlsx')
zhu_ti_nei_rong = pd.read_excel(r'.\汽车数据\主题内容.xlsx')
zhu_ti_nei_rong.head()
test = wen_zhang_nei_rong.head(5000)
test = wen_zhang_xing_xi.head(6000)
wen_zhang_nei_rong['文章ID']=wen_zhang_nei_rong['文章ID'].astype('object')
wenzhang = pd.merge(wen_zhang_nei_rong,wen_zhang_xing_xi,on='文章ID')
wenzhang.info()
set(wen_zhang_nei_rong['文章ID']) - set(wen_zhang_xing_xi['文章ID'])


che_xi_biao.info()
che_xi_biao['车系ID'] = che_xi_biao['车系ID'].astype('object')
wenzhang = pd.merge(che_xi_biao,wenzhang,how='right',on='车系ID')

zhu_ti_xing_xi.info()
zhu_ti_nei_rong.info()

zhuti = pd.merge(zhu_ti_xing_xi,zhu_ti_nei_rong,how='left',on='主题ID')
zhuti = pd.merge(che_xi_biao,zhuti,how='right',on='车系ID')
test = zhuti.head(10000)
test = zhu_ti_xing_xi.head(1000)
test.info()

