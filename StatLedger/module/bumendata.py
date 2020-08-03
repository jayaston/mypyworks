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
        
        self.engine = create_engine("mysql+pymysql://"+str(user)+":"+str(password)+"@"+str(host)+":"+str(port)+"/"+str(database)+"?charset=utf8",
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
        self.conn.close()#关闭连接
        return(df_result)
    def get_all_quota(self):
        sql = "SELECT QUOTA_CODE,QUOTA_NAME\
            FROM TJFX.CS_QUOTA_DEFINE"
        df_zhibiao = pd.read_sql(sql,self.conn) 
        self.conn.close() 
        result = df_zhibiao.drop_duplicates(['QUOTA_CODE'])
        return result
    
    def get_all_dept(self):
        sql = "SELECT DISTINCT GROUP_CODE , GROUP_NAME\
            FROM TJFX.HR_ORGANIZATION"  
        df_bumen = pd.read_sql(sql,self.conn) 
        self.conn.close()
        result = df_bumen.drop_duplicates(['GROUP_CODE'])
        return result
    def get_formula(self):
        sql = "SELECT i.TZ_TYPE ,r.QUOTA_CODE, q.QUOTA_NAME,r.ZB_DEPT_CODE,o.GROUP_NAME,r.FORMULA_CODE,f.FORMULA,f.START_TIME,f.END_TIME,v.VIEW_NAME 方案,i.TZ_NAME as 目录\
            FROM CS_TZZB_RELATION r LEFT JOIN CS_TZ_VIEW v ON r.VIEW_CODE = v.VIEW_CODE\
						LEFT JOIN CS_TZ_ITEM i ON r.TZ_CODE = i.TZ_CODE\
						LEFT JOIN cs_formula_set f ON r.FORMULA_CODE = f.FORMULA_CODE\
						LEFT JOIN CS_QUOTA_DEFINE q ON r.QUOTA_CODE = q.QUOTA_CODE\
						LEFT JOIN HR_ORGANIZATION o ON r.ZB_DEPT_CODE = o.GROUP_CODE\
            WHERE  r.FORMULA_CODE is not null\
            AND v.VIEW_NAME IN ('新系统','计划组','各区所去年发单水量','东区','中区','南区','北区','财务部','水表厂','水质部')"  
        df_formula = pd.read_sql(sql,self.conn) 
        self.conn.close()
        result = df_formula.drop_duplicates(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE','FORMULA_CODE'])
        return result
    def get_formula_table(self):
        sql = "select * from FORMULA"  
        result = pd.read_sql(sql,self.conn) 
        self.conn.close()        
        return result
        
        
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
            print("导入失败，建立临时表错误！")
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
                sql = "INSERT INTO tj_data SELECT\
                        QUOTA_CODE,\
                        MON,\
                        str_to_date( QUOTA_DATE, '%Y-%m-%d %H:%i:%s' ) AS QUOTA_DATE,\
                        QUOTA_VALUE,\
                        REPORT_FLAG,\
                        QUOTA_DEPT_CODE,\
                        IMPORT_FLOW_NO,\
                        WARNING_CODE,\
                        RECORD_TYPE \
                        FROM\
                        	TJ_DATA_TMP3 \
                        	ON DUPLICATE KEY UPDATE QUOTA_VALUE =\
                        VALUES\
                        	( QUOTA_VALUE ),\
                        	REPORT_FLAG =\
                        VALUES\
                        	( REPORT_FLAG ),\
                        	IMPORT_FLOW_NO =\
                        VALUES\
                        	( IMPORT_FLOW_NO ),\
                        	WARNING_CODE =\
                        VALUES\
                        	( WARNING_CODE )"
                try:
                    c.execute(sql)#执行sql语句
                    self.conn.commit()#提交到数据库执行
                    print('数据上传完毕！')   
                except:
                    print ('导入失败，写入主数据错误')
                    traceback.print_exc()
                    self.conn.rollback()#发生错误时，回滚！
        c.close()  # 关闭游标
        self.conn.close()#关闭连接
    def imp_tbl(self,df,tblname):
        def mapping_df_types(df):
            dtypedict = {}
            for i, j in zip(df.columns, df.dtypes):
                if "object" in str(j):
                    dtypedict.update({i: VARCHAR(800)})
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
        df.to_sql(tblname, con=self.engine,if_exists = 'replace',index=False, dtype=dtypedict)
        self.conn.close()#关闭连接
    
    def update_tbl(self,df,
                   sql:str="INSERT INTO `stock_discover` \
                       VALUES (%s, %s, %s, %s, %s, %s)\
                  ON DUPLICATE KEY UPDATE `date` = VALUES(`date`) , yesterday = VALUES(yesterday)"):
        c = self.conn.cursor()
        #数据格式如下：
        mylist = df.values.tolist()
        #批量插入使用executement
        c.executemany(sql,mylist)        
        self.conn.commit()
        c.close()
        self.conn.close()#关闭连接

        
if __name__ == "__main__":
    
    df2 = BumenData().getdata('20181231', '20190101')

    