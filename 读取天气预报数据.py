# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:26:39 2020

@author: XieJie
"""

 
    
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