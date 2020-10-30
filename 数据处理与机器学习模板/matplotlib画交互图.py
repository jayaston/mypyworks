# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:20:02 2020

@author: XieJie
"""


import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib as mpl  # 导入中文字体，避免显示乱码
import os
#%matplotlib

mpl.rcParams['font.sans-serif'] = ['SimHei'] #用来显示中文，防止乱码
mpl.rcParams['font.family'] = 'sans-serif' #用来显示中文，防止乱码
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

os.getcwd() 

#获取数据
file = r'c:\Users\XieJie\mypyworks\data.xlsx' #Excel文件的位置，更换为自己的位置就行
url_list = [r'https://www.boxofficemojo.com/date',
                r'https://www.boxofficemojo.com/date/2020-02-24',
                r'https://www.boxofficemojo.com/date/2020-02-25',
                r'https://www.boxofficemojo.com/date/2020-02-26',
                r'https://www.boxofficemojo.com/date/2020-02-27',
                r'https://www.boxofficemojo.com/date/2020-02-28',
                r'https://www.boxofficemojo.com/date/2020-02-29',
                r'https://www.boxofficemojo.com/date/2020-03-01'
]
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'text/plain;charset=UTF-8'}        
sheet_names = ['overview', '02-24', '02-25', '02-26', '02-27', '02-28', '02-29', '03-01']
writer = pd.ExcelWriter(file, engine='openpyxl')
for i in range(len(url_list)):        
    res = requests.get(url_list[i], headers=headers)
    text =res.text
    table = pd.read_html(text)
    table = table[0]
    table.to_excel(writer, sheet_name=sheet_names[i])    
writer.save()
writer.close()


def click(event):
    if event.artist != art:
        return 
    ind = event.ind[0]
    day_data = data_list[ind]
    titles = title_list[ind]
    ax1.cla()
    ax1.barh(range(len(day_data)), day_data)
    ax1.set_yticklabels(titles)
    ax1.set_xlabel('票房/万美元')
    ax1.set_yticks(range(len(day_data)))
    ax.set_title('This is the top10 of %s' % sheet_names[1:][ind])
    fig.canvas.draw()
   
        
overview_table = pd.read_excel(file, sheet_name='overview')
overview_data = overview_table['Top 10 Gross'][1:8]
overview_data = overview_data.str.replace('$', '') 
overview_data = overview_data.str.replace(',', '') 
overview_data = overview_data.astype(int, copy=False)/100000
overview_data = overview_data.round(2)
overview_data = overview_data[::-1].to_list()

data_list = []
title_list = []
for date in sheet_names[1:]:
    day_table = pd.read_excel(file, sheet_name=date)
    day_data = day_table['Daily'][:10]
    day_data = day_data.str.replace('$', '') 
    day_data = day_data.str.replace(',', '') 
    day_data = day_data.astype(int, copy=False)/10000
    day_data = day_data.round(2)
    day_data = day_data.to_list()
    day_title = day_table['Release'][:10]
    day_title = day_title.to_list()
    data_list.append(day_data)
    title_list.append(day_title)

fig, (ax, ax1) = plt.subplots(2, 1)
ax.set_xticks(range(len(overview_data)))
ax.set_ylim(0,400)
ax.set_title('click on point to plot time series')
ax.set_ylabel('票房/十万美元')
art, = ax.plot(overview_data, marker='o', picker=5)
ax.set_xticklabels(sheet_names[1:])
plt.subplots_adjust(left=0.3, hspace=0.4)
fig.canvas.mpl_connect('pick_event', click)
plt.show()

