# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:46:59 2021

@author: xiejie
"""
import os
os.getcwd()
import pandas as pd
from win32com import client as wc
from pydocx import PyDocX
from  bs4 import BeautifulSoup
def doSaveAas(doc_path):
        """
        将doc文档转换为docx文档

        :rtype: object
        """

        docx_path = doc_path.replace("doc", "docx")
        word = wc.Dispatch('Word.Application')
        doc = word.Documents.Open(doc_path)  # 目标路径下的文件
        doc.SaveAs(docx_path, 12, False, "", True, "", False, False, False, False)  # 转化后路径下的文件
        doc.Close()
        word.Quit()
startmon = '202112' 
name = "20211221"        
doc_path="C:\\Users\\xiejie/mypyworks/自来水数据/天气预测网页数据/"+name+".doc"
docx_path = doc_path.replace("doc", "docx")

doSaveAas(doc_path)
response = PyDocX.to_html(docx_path)

#response[response.find('二十')-100:response.find('二十')+1000]

soup = BeautifulSoup(response,"html.parser")





#最高温
maxtemp=soup.findAll("span",style="color:#006699")
maxtemp=maxtemp[::2 ] #获取奇数
maxtemp = [int(x.string[:-1]) for x in maxtemp]

#最低温
mintemp = soup.findAll("span",style="color:#6EAFD7")
mintemp=mintemp[::2 ] #获取奇数
mintemp = mintemp[-len(maxtemp):]
mintemp = [int(x.string[:-1]) for x in mintemp]
#天气
tianqi = soup.findAll("span",style="color:#8E97A3")
tianqi = tianqi[-len(maxtemp)*9:][3::9]
tianqi = [str(x.string) for x in tianqi]
#预测开始日期
riqi = soup.findAll("h2")[-len(maxtemp):]
riqi = [x.string[-2:] for x in riqi][0]
riqi = pd.date_range(start=startmon+riqi,periods=len(maxtemp)).strftime('%Y%m%d')


df = pd.DataFrame(data=list(zip(riqi,maxtemp,mintemp,tianqi)),columns=["日期","最高温","最低温","天气"])
df.to_excel("~/mypyworks/自来水数据/天气预测网页数据/"+name+".xls")
