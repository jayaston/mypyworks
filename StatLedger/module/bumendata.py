# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:58:56 2019

@author: XieJie
"""
import pandas as pd
import pymysql
import traceback
import readConfig
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import \
            FLOAT,VARCHAR,INTEGER,DATETIME,BOOLEAN

class BumenData:
    def __init__(self,
                 host=readConfig.bumen_host,
                 port=readConfig.bumen_port,
                 database=readConfig.bumen_database,
                 user=readConfig.bumen_user,
                 password=readConfig.bumen_pwd,
                 charset='utf8'):
        self.conn = pymysql.connect(host=host,port=port,database=database,
                               user=user,password=password,charset=charset)
        
        self.engine = create_engine("mysql+pymysql://"+user+":"+password+"@"+host+":"+port+"/"+database+"?charset=utf8",
                                    echo=False)
    def getdata(self,startd,endd,quotas=None):        
        try:
            sql = "select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
            from tj_data \
            where date_format(quota_date,'%Y%m%d') >= '"+startd+"' \
            and date_format(quota_date,'%Y%m%d') <='"+endd+"'"
            df = pd.read_sql(sql,self.conn)
            df_result = pd.DataFrame()       
            for item in quotas:                        
                df_result_new =df[(df.QUOTA_DEPT_CODE==item[0]) & (df.QUOTA_CODE==item[1]) & (df.RECORD_TYPE==item[2])]
                df_result = pd.concat([df_result,df_result_new])
        except Exception as e:
            print("%s;没有指定具体指标，返回全部数据"%e)
            df_result = df        
        print(df_result.head())        
        return(df_result)
        
        
    def importdata(self,mylist:list):
        c = self.conn.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor
        try:
            sql = "drop table TJ_DATA_TMP3"
            c.execute(sql)#删除已经存在的临时表            
        except:            
            pass
            
        sql="CREATE TABLE TJ_DATA_TMP3 ( \
            QUOTA_CODE VARCHAR ( 12 ), \
            MON VARCHAR ( 6 ), \
            QUOTA_DATE VARCHAR ( 20 ), \
            QUOTA_VALUE VARCHAR ( 64 ), \
            REPORT_FLAG CHAR ( 1 ), \
            QUOTA_DEPT_CODE VARCHAR ( 36 ), \
            IMPORT_FLOW_NO VARCHAR ( 12 ), \
            WARNING_CODE VARCHAR ( 12 ), \
            RECORD_TYPE CHAR ( 1 ), \
            CONSTRAINT pk_tmp3 PRIMARY KEY ( QUOTA_CODE, QUOTA_DATE, QUOTA_DEPT_CODE, RECORD_TYPE ))"
            
        try:
            c.execute(sql)       #删除已经存在的临时表
        except:            
            print("导入失败，建立零时表错误！")
            traceback.print_exc()

        else:
            sql = "INSERT INTO TJ_DATA_TMP3 \
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    
            try:
                c.executemany(sql,mylist)#执行sql语句
                self.conn.commit()#提交到数据库执行
            except:
                print ('导入失败，报表数据导入错误！')
                traceback.print_exc()
                self.conn.rollback()#发生错误时，回滚！
            else:
                sql = "INSERT INTO tj_data ( QUOTA_DATE, QUOTA_DEPT_CODE, QUOTA_CODE, QUOTA_VALUE, RECORD_TYPE ) SELECT \
                str_to_date(QUOTA_DATE,'%Y-%m-%d %H:%i:%s') as QUOTA_DATE, \
                QUOTA_DEPT_CODE, \
                QUOTA_CODE, \
                QUOTA_VALUE, \
                RECORD_TYPE  \
                FROM \
                	TJ_DATA_TMP3 t \
                	ON DUPLICATE KEY UPDATE QUOTA_VALUE = t.QUOTA_VALUE"
                try:
                    c.execute(sql)#执行sql语句
                    self.conn.commit()#提交到数据库执行
                    print('数据上传完毕！')   
                except:
                    print ('导入失败，写入主数据错误')
                    traceback.print_exc()
                    self.conn.rollback()#发生错误时，回滚！
        c.close()  # 关闭游标
    def imp_tbl(self,df,tblname):
        def mapping_df_types(df):
            dtypedict = {}
            for i, j in zip(df.columns, df.dtypes):
                if "object" in str(j):
                    dtypedict.update({i: VARCHAR(255)})
                if "float" in str(j):
                    dtypedict.update({i: FLOAT()})
                if "int" in str(j):
                    dtypedict.update({i: INTEGER()})
                if "datetime" in str(j):
                    dtypedict.update({i: DATETIME()})
                if "bool" in str(j):
                    dtypedict.update({i: BOOLEAN()})
            return dtypedict        
        dtypedict = mapping_df_types(df)
        df.to_sql(tblname, con=self.engine,if_exists = 'replace',schema='tjdata',index=False, dtype=dtypedict)

    
    def update_tbl(self,df,
                   sql:str="INSERT INTO `stock_discover` VALUES (%s, %s, %s, %s, %s, %s)\
                  ON DUPLICATE KEY UPDATE `date` = VALUES(`date`) , yesterday = VALUES(yesterday)"):
        c = self.conn.cursor()
        #数据格式如下：
        mylist = list(df.to_records(index=False))
        #批量插入使用executement
        c.executemany(sql,mylist)        
        self.conn.commit()
        c.close()
    def close(self):
        self.conn.close()
        
if __name__ == "__main__":
    b = BumenData()
    df2 = b.getdata('20181231', '20190101')
    b.close()
    