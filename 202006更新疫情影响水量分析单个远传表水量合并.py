# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:49:36 2020

@author: Jay
"""

import pandas as pd
df_kfq = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(900051635206)广州经济技术开发区供水管理中心2018-05-01到2020-06-28历史数据.xls')
df_hd = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999030320201)广州市恒大房地产开发有限公司2018-05-01到2020-06-28历史数据.xls')
df_ld = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999051300206)天河区沙河镇龙洞村委会2018-05-01到2020-06-28历史数据.xls')
df_xjc = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999810000610)新机场2018-05-01到2020-06-28历史数据.xls')
df_hg = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999810001262)华南工学院2018-05-01到2020-06-28历史数据.xls')
df_jta = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999810001353)姬堂村A表2018-05-01到2020-06-28历史数据.xls')
df_jtb = pd.read_excel(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\表(999810001354)姬堂村B表2018-05-01到2020-06-28历史数据.xls')

#合并
df = pd.concat([df_kfq,df_hd,df_ld,df_xjc,df_hg,df_jta,df_jtb])
df.rename(columns={'时间':'datatime','表码':'meterid','行度值':'datavalue'},inplace=True)
df.to_csv(r'c:\Users\Jay\Desktop\模仿E20网站新冠疫情影响供水行业分析\远传表数据\合并表.csv',index=0)