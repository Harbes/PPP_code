#import pymysql
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
def WriteToMySQL(data,table_name,db_name):
    engine = create_engine('mysql+mysqlconnector://root:1234567890@localhost:3306/'+db_name)
    data.to_sql(name=table_name,con=engine,index=False,if_exists='append')
    return None

bs=pd.read_sql('select * from AShareFinancialIndicator',connect_mysql) #
bs.to_pickle(data_path_local+'AShareFinancialIndicator')



import statsmodels.api as sm
Y = [1,3,4,5,2,3,4]*100
X = range(1,701)

X = sm.add_constant(X)
res = sm.OLS(Y,X).fit()
res.params


# todo ashareEODfinancialIndicator 中常用的数据保存
market_cap = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_mv,s_dq_mv', 'AShareEODDerivativeIndicator')
col_names='trade_dt,s_info_windcode,s_val_mv,s_dq_mv'
table_name='AShareEODDerivativeIndicator'
tmp=pd.read_sql(OrderToGetColumnsFromSQL(col_names, table_name), connect_mysql)
tmp.to_pickle(data_path_local+'AShareEODDerivativeIndicator')
tmp=pd.read_pickle(data_path_local+'AShareEODDerivativeIndicator')
market_cap = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_mv', 'AShareEODDerivativeIndicator')
market_cap.tail()
market_cap.to_pickle(data_path_local+'market_cap_free')

# 证监会行业分类
sql_order='SELECT DISTINCT(SEC_IND_CODE) from asharesecnindustriesclass'
ind=pd.read_sql(sql_order,connect_mysql)
ind['1st']=ind['SEC_IND_CODE'].map(lambda x:str(x)[2:4])
ind['2nd']=ind['SEC_IND_CODE'].map(lambda x:str(x)[4:6])
ind['ind']=ind['1st']+ind['2nd']


trading_calendar.max()
sec_ind_1st=pd.DataFrame(np.nan,index=trading_calendar,columns=ashare_stock)
sec_ind_2nd=pd.DataFrame(np.nan,index=trading_calendar,columns=ashare_stock)
sql_order='SELECT s_info_windcode,SEC_IND_CODE,entry_dt,remove_dt from asharesecnindustriesclass'
ind=pd.read_sql(sql_order,connect_mysql)
ind['1st']=ind['SEC_IND_CODE'].map(lambda x:str(x)[2:4])
ind['2nd']=ind['SEC_IND_CODE'].map(lambda x:str(x)[4:6])
ind['1st_2nd']=ind['1st']+ind['2nd']
ind['1st+2nd']=ind['1st'].where(ind['1st']!='03',ind['1st_2nd'])
ind['entry_dt'].loc[ind['entry_dt'].isnull()]=ind['remove_dt'].loc[ind['entry_dt'].isnull()]
ind.loc[[1845,1885,3705,4231],'remove_dt']=None;ind.loc[[1845,1885,3705,4231]]
ind['entry_dt']=pd.to_datetime(ind['entry_dt'],format='%Y%m%d')
ind['remove_dt'].loc[ind['remove_dt'].isnull()]='20191210'
ind['remove_dt']=pd.to_datetime(ind['remove_dt'],format='%Y%m%d')
ind_1st=ind[['s_info_windcode','entry_dt','1st']].set_index(['entry_dt','s_info_windcode']).sort_index().unstack().resample('D').ffill().ffill()
ind_1st.columns=ind_1st.columns.droplevel()
#ind_1st.reindex(index=trading_calendar,columns=ashare_stock,method='ffill').to_pickle(data_path_local+'industry_sec_1st')
ind_2nd=ind[['s_info_windcode','entry_dt','1st_2nd']].set_index(['entry_dt','s_info_windcode']).sort_index().unstack().resample('D').ffill().ffill()
ind_2nd.columns=ind_2nd.columns.droplevel()
#ind_2nd.reindex(index=trading_calendar,columns=ashare_stock,method='ffill').to_pickle(data_path_local+'industry_sec_2nd')
ind_all=ind[['s_info_windcode','entry_dt','SEC_IND_CODE']].set_index(['entry_dt','s_info_windcode']).sort_index().unstack().resample('D').ffill().ffill()
ind_all.columns=ind_all.columns.droplevel()
#ind_all.reindex(index=trading_calendar,columns=ashare_stock,method='ffill').to_pickle(data_path_local+'industry_sec_all')
ind_1st_2nd=ind[['s_info_windcode','entry_dt','1st+2nd']].set_index(['entry_dt','s_info_windcode']).sort_index().unstack().resample('D').ffill().ffill()
ind_1st_2nd.columns=ind_1st_2nd.columns.droplevel()
ind_1st_2nd.reindex(index=trading_calendar,columns=ashare_stock,method='ffill').to_pickle(data_path_local+'industry_sec_1st_2nd')


sql_order='select s_ipo_listdate,s_info_windcode from ashareipo'
IPO=pd.read_sql(sql_order,connect_mysql)
IPO.loc[IPO['s_ipo_listdate'].isnull(),'s_ipo_listdate']='20191210'
IPO.loc[IPO['s_ipo_listdate'].isnull()]
IPO['s_ipo_listdate']=pd.to_datetime(IPO['s_ipo_listdate'],format='%Y%m%d')
IPO['mark']=1.0
IPO=IPO.set_index(['s_ipo_listdate','s_info_windcode']).unstack()
IPO.columns=IPO.columns.droplevel()
IPO.reindex(index=trading_calendar,columns=ashare_stock)#.to_pickle(data_path_local+'IPO')



import numpy as np
import pandas as pd
T=30;
N=1000
X_=pd.DataFrame(np.random.randn(T,3))
Y_=pd.DataFrame(np.random.randn(T,N))
Y_-X_@np.linalg.pinv(X_.T@X_)@X_.T@Y_


data_path_local = 'E:/data/winddb/'
freq_list = ['M', '2W', 'W']
freq=freq_list[2]
predictors_all=pd.read_pickle(data_path_local + 'predictors_all_' + freq)
predictors_all.columns=predictors_all.columns.str.strip()
predictors_all.to_pickle(data_path_local + 'predictors_all_' + freq+'_columsn_adjusted')



a=np.random.randn(300,5)
aa=a+np.array([1,2,3,4,5]*300).reshape(300,5)
np.corrcoef(aa.T)
np.corrcoef(a.T)
b,c=np.linalg.eigh(np.corrcoef(a.T))
np.diag(b)-c.T@np.corrcoef(a.T)@c
np.corrcoef((a@c).T)



interest.head()
