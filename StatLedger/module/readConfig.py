# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:32:48 2019

@author: Jay
"""
import os
import configparser
#获取配置文件目录
dir_path =  os.path.dirname(os.path.dirname(__file__))
configPath = os.path.abspath(os.path.join(dir_path,"数据表/SQlconfig.ini"))    

#读取配置文件信息
conf = configparser.ConfigParser()
conf.read(configPath)

#读取配置文件参数
tjfx_user = conf.get("tjfxdata","user")
tjfx_pwd = conf.get("tjfxdata","pwd")
tjfx_dsn = conf.get("tjfxdata","dsn")

bumen_user = conf.get("bumendata","user")
bumen_pwd = conf.get("bumendata","pwd")
bumen_host = conf.get("bumendata","host")
bumen_port = int(conf.get("bumendata","port"))
bumen_database = conf.get("bumendata","database")
