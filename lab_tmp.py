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


if __name__=='__main__':
    data_path_local = 'E:/data/winddb/'
    options_mysql={'server':'localhost',
                'user':'root',
                'password':'1234567890',
                'database':'winddb'}
    connect_mysql=ConnectMySQL(**options_mysql)
    data=pd.read_sql('select S_INFO_WINDCODE,TRADE_DT,S_DQ_VOLUME,S_DQ_AMOUNT from AShareEODPrices',connect_mysql)
    data.to_pickle(data_path_local+'AShareEODPrices')
    #tmp=pd.read_sql('select * from ashareeodprices LIMIT 10',connect_mysql)
    # 'S_INFO_WINDCODE,TRADE_DT,S_DQ_PRECLOSE,S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW,S_DQ_CLOSE,S_DQ_VOLUME,S_DQ_AMOUNT,S_DQ_ADJPRECLOSE,S_DQ_ADJOPEN,S_DQ_ADJCLOSE'