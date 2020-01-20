import numpy as np
import pandas as pd
import pymysql
def ConnectMySQL(server,user,password,database):
    '''
    范例：
    options_WindDatabase={'server':'localhost',
                      'user':'root',
                      'password':'123456',
                      'database':'winddb'}
    connect=ConnectMySQL(**options_mysql)
    '''
    connect=pymysql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect
options_mysql={'server':'localhost',
                'user':'root',
                'password':'1234567890',
                'database':'winddb'}
connect_mysql=ConnectMySQL(**options_mysql)

def GetColumnsFromSQL(cols,table):
    return "select "+cols+' from '+table

# 变量数据