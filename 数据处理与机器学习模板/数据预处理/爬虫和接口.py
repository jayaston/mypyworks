# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:46:01 2020

@author: XieJie
"""

import requests
import pandas as pd
import time

#读取网页数据
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

#读取接口数据
import urllib.request,urllib.parse
import ssl
import json
host = "http://aliv8.data.moji.com"
path = "/whapi/json/aliweather/forecast15days"
method = 'POST'
appcode = '43fca530b164413d82b5af24000a56e5'
querys = ''
bodys = {}
url = host + path

bodys['lat'] = "23.13581"
bodys['lon'] = "113.31935"
bodys['token'] = "0f9d7e535dfbfad15b8fd2a84fee3e36"
post_data = bytes(urllib.parse.urlencode(bodys), encoding='utf8')
request = urllib.request.Request(url,data = post_data, method=method)
request.add_header('Authorization', 'APPCODE ' + appcode)
#根据API的要求，定义相对应的Content-Type
request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
response = urllib.request.urlopen(request,context=ctx)
content = response.read().decode('utf-8')
if (content):
    print(json.dumps(eval(content),ensure_ascii=False,indent=4))