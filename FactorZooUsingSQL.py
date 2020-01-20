# 利用本地wind数据库数据计算Factors
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from pandas.tseries.offsets import MonthEnd,YearEnd,Week,Day,DateOffset
import statsmodels.api as sm
from scipy import stats
#from TestFactor import *

# 全局变量设置
#qcut_options = {'q': np.arange(0.0, 1.001, 0.2)};
#qcut_options['labels'] = list(range(1, len(qcut_options['q'])))
## 链接数据库
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
data_path_local='E:/data/winddb/'

## 可重复工具箱
def GenerateErrorFromOLSbyMatrix(Y,X):
    Y_=(Y-Y.mean()).fillna(0.0)
    X_=pd.DataFrame((X-X.mean()).fillna(0.0))
    return Y_-X_.values@np.linalg.pinv(X_.T@X_)@(X_.T@Y_).values
    #return res#[res!=0.0]
def WriteToMySQL(data,table_name,db_name,write_index=True,if_exists='append'):
    engine = create_engine('mysql+mysqlconnector://root:1234567890@localhost:3306/'+db_name)
    data.to_sql(name=table_name,con=engine,index=write_index,if_exists=if_exists)
    return None
def OrderToGetColumnsFromSQL(cols, table):
    # conditions 的输入请参照以下格式：[不推荐添加conditions，速度太慢！！！]
    #   'trade_dt between 19980101 and 20191201' 或者：
    #   '(trade_dt between 19980101 and 20191201) and (s_dq_turn != 0)'
    return "select "+cols+' from '+table
def GenerateDataframeFromWindDBorLocal(col_names, table_name, source='local'):
    col_names=col_names.upper()
    index_by=str.split(col_names,',')[:2]
    if source=='local':
        tmp=pd.read_pickle(data_path_local+table_name)[col_names.split(',')].set_index(index_by)
    else:
        tmp=pd.read_sql(OrderToGetColumnsFromSQL(col_names, table_name), connect_mysql).set_index(index_by)
    tmp=tmp.loc[~tmp.index.duplicated(keep='last')].unstack()
    tmp.columns=tmp.columns.droplevel()
    tmp.index=pd.to_datetime(tmp.index,format='%Y%m%d')
    return tmp.astype(float)
def GetYearendDataFromDaily(df):
    ## todo 待测试
    '''主要针对的是类似量价数据，而非财报数据'''
    df=df.loc[df.index.month==12].groupby(lambda x:x.year).last()
    df.index=pd.to_datetime(df.index,format='%Y')+YearEnd()
    return df
def GetInterestRateFromLocal(date_investing,freq='M'):
    interest = pd.read_excel(data_path_local + 'interest_rate.xlsx')
    if freq.upper()=='M':
        return interest[['Clsdt','Nrrmtdt']].set_index('Clsdt').sort_index()['Nrrmtdt'].reindex(date_investing)
    elif freq.upper()=='2W':
        return interest[['Clsdt', 'Nrrmtdt']].set_index('Clsdt').sort_index()['Nrrmtdt'].reindex(date_investing)*0.5
    else:
        return interest[['Clsdt', 'Nrrwkdt']].set_index('Clsdt').sort_index()['Nrrwkdt'].reindex(date_investing)
# 分析工具箱
def Returns_UnivariatePortfolio(chara,ret,qcut_options,weights=None):
    '''
    Given the aligned pairs (chara,ret), generate the characteristic-based portfolio returns
    :param chara: characteristic ; DataFrame[T*N]
    :param ret: stock returns; DataFrame[T*N] , 与chara数据格式匹配
    :param qcut_options: options setting for pd.qcut; 'q'、'labels'
    :param weights: weights for portfolio construction,if None, then equal weights.
    :return: returns for group portfolios
    '''
    if chara.shape != ret.reindex(chara).shape:
        raise ValueError('chara和ret数据没有对齐！')
    #ValidData = (~chara.isnull()) & (~ret.isnull())
    if weights is not None and (chara.shape != weights.shape):
        raise ValueError('weights与其他数据没有对齐！')
        #else:
        #    ValidData= ValidData & (~weights.isnull())
        #    weights=weights[ValidData]
    #chara = chara[ValidData]
    #ret = ret[ValidData]
    #mark_group=pd.DataFrame(np.nan,index=chara.index,columns=chara.columns)
    mark_group = chara.dropna(axis=0,how='all').apply(lambda x: pd.qcut(x, q=qcut_options['q'], labels=qcut_options['labels']), axis=1)
    ret_portfolio=pd.DataFrame(np.nan,index=mark_group.index,columns=qcut_options['labels'])
    if weights is None:
        for c in qcut_options['labels']:
            ret_portfolio.loc[:,c]=ret[mark_group==c].mean(axis=1)
    else:
        agg_weights=pd.DataFrame(np.nan,index=chara.index,columns=qcut_options['labels'])
        ret_weights = ret * weights
        weights_ = (~ret.isnull()) * weights
        for c in qcut_options['labels']:
            ret_portfolio.loc[:,c]=ret_weights[mark_group==c].sum(axis=1)
            agg_weights.loc[:,c]=weights_[mark_group==c].sum(axis=1)
        ret_portfolio=ret_portfolio/agg_weights
    return ret_portfolio.astype(float)
def Returns_BivariatePortfolio(chara,chara_con,ret,qcut_options,weights=None,independent=False):
    '''
    :param chara:
    :param chara_con: conditional characteristic ; DataFrame[T*N]
    :param ret: 与chara等匹配的格式
    :param qcut_options: 'q','q_con','labels','labels_con'
    :param weights:
    :return:
    '''

    if (chara.shape != ret.reindex(chara).shape) or (chara_con.shape != ret.reindex(chara).shape):
        raise ValueError('chara、chara_con和ret数据没有对齐！')
    if (weights is not None) and (chara.shape != weights.shape):
        raise ValueError('weights与其他数据没有对齐！')
    mark_group=pd.DataFrame(np.nan,index=chara.index,columns=chara.columns)
    mark_group_con=pd.DataFrame(np.nan,index=chara_con.index,columns=chara_con.columns)
    if independent:
        mark_group=chara.apply(lambda x:pd.qcut(x,q=qcut_options['q'],labels=qcut_options['labels']),axis=1)
        mark_group_con = chara_con.apply(lambda x: pd.qcut(x, q=qcut_options['q_con'],labels=qcut_options['labels_con']), axis=1)
        #for t in chara.index:
        #    mark_group.loc[t]=pd.qcut(chara.loc[t],q=qcut_options['q'],labels=qcut_options['labels'])
        #    mark_group_con.loc[t]=pd.qcut(chara_con.loc[t],q=qcut_options['q_con'],labels=qcut_options['labels_con'])
    else:
        #
        mark_group_con = chara_con.apply(lambda x: pd.qcut(x, q=qcut_options['q_con'],labels=qcut_options['labels_con']), axis=1)
        #for t in chara_con.index:
        #    mark_group_con.loc[t] = pd.qcut(chara_con.loc[t], q=qcut_options['q_con'],
        #                                    labels=qcut_options['labels_con'])
        for i in qcut_options['labels_con']:
            mark_group=mark_group.combine_first(chara[mark_group_con==i].apply(lambda x:pd.qcut(x,
                                                                                  q=qcut_options['q'],
                                                                                  labels=qcut_options['labels']),axis=1))
    ret_portfolio = pd.DataFrame(np.nan, index=pd.MultiIndex.from_product((chara.index,qcut_options['labels_con'])),
                                 columns=qcut_options['labels'])

    if weights is None:
        for i in qcut_options['labels_con']:
            for c in qcut_options['labels']:
                ret_portfolio.loc[(slice(None),i),c]=\
                    ret[(mark_group_con==i) & (mark_group==c)].mean(axis=1).values
    else:
        agg_weights = pd.DataFrame(np.nan, index=pd.MultiIndex.from_product([chara.index, qcut_options['labels_con']]),
                                     columns=qcut_options['labels'])
        ret_weights=ret*weights
        weights_=(~ret.isnull()) * weights
        for i in qcut_options['labels_con']:
            for c in qcut_options['labels']:
                ret_portfolio.loc[(slice(None),i),c]=ret_weights[(mark_group_con==i) & (mark_group==c)].sum(axis=1)
                agg_weights.loc[(slice(None),i),c]=weights_[(mark_group_con==i) & (mark_group==c)].sum(axis=1)
        ret_portfolio=ret_portfolio/agg_weights
    return ret_portfolio.astype(float)
def DescribeDataframe(df,stats=['skew','kurt']):
    d=df.describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
    return d.append(df.reindex(d.columns, axis=1).agg(stats))
def NWest_for_mean(dataframe,L=None):
    '''
    df不要以Nan开头，会引起误差;
    求时序均值的标准差估计
    :param df:
    :param L:
    :return:
    '''
    df=dataframe-dataframe.mean()
    T=len(df)
    if L is None:
        L=int(T**0.25)
    w=1.0-np.arange(1,L+1)/(L+1.0)
    return np.sqrt(2.0*pd.DataFrame((df*df.shift(i+1)*w[i]).sum() for i in range(L)).sum()/T+df.var())/np.sqrt(T)
def NWest(e,X,L=None):
    '''
    输入（残差序列+constant）序列以及解释变量X，返回t数值
    '''
    T = len(e)
    if L is None:
        L = int(T ** 0.25) # or : L = 0.75*T**(1.0/3.0)-1
    w = 1.0 - np.arange(1, L + 1) / (L+1.0)
    X.insert(0,'c',np.ones(T))
    S=0.0
    for l in range(1,L+1):
        for i in range(l,T):
            S+=w[l-1]*e[i]*e[i-l]*(X.iloc[i][:,None]*X.iloc[i-l].values+X.iloc[i-l][:,None]*X.iloc[i].values)
    for i in range(T):
        S+=e[i]*e[i]*X.iloc[i][:,None]*X.iloc[i].values
    XX_1=np.linalg.pinv(X.T@X.values)
    X.drop('c', axis=1, inplace=True)
    return np.sqrt((XX_1@S@XX_1)[0,0])
def PortfolioAnalysis(df,winso=False):
    # 给定一个因子收益率的dataframe
    # todo 输出 均值收益率，t数值，SR，std等
    if winso:
        df_ = pd.DataFrame(df.dropna(how='all', axis=0)).apply(lambda y:stats.mstats.winsorize(y,limits=[0.01,0.01]),axis=0)
    else:
        df_=pd.DataFrame(df.dropna(how='all',axis=0))
    res=pd.DataFrame(np.nan,index=['mean','std','SR','SR_ann','t','t_NW_adjusted'],columns=df_.columns)
    res.loc['mean']=df_.mean()
    res.loc['std']=df_.std()
    res.loc['SR']=res.loc['mean']/res.loc['std']
    res.loc['t']=res.loc['SR']*np.sqrt(len(df_))
    res.loc['t_NW_adjusted']=res.loc['mean']/NWest_for_mean(df_) # return的量级影响t_NW_adjusted
    res.loc['SR_ann']=res.loc['mean']*12.0/AnnualizeStd(df_)
    return res
def AnnualizeStd(df):
    rho=pd.DataFrame(np.nan,index=np.arange(1,12),columns=df.columns)
    rho=rho.apply(lambda x:df.apply(lambda y:y.autocorr(x.name)),axis=1)
    return df.std()*np.sqrt(12.0+2.0*rho.mul(np.arange(11,0,-1),axis=0).sum())
# 股票筛选
def GetNotTradingFromSQL(source='local'):
    if source == 'local':
        return pd.read_pickle(data_path_local+'NT_30')
        #(1.0*(NT == 0.0)).fillna(0.0).to_pickle(data_path_local+'NT_by_TradeVol')
    else:
        # limit=110 删掉了太多的 2006-07的数据
        NT= GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
        return (1.0*(NT.reindex(trading_calendar)==0.0)).replace(0.0,np.nan).ffill(limit=30).isnull()
def GetSTFromSQL(ashare_stock=None,source='local'):

    if source=='local':
        return pd.read_pickle(data_path_local+'ST')
    else:
        #trading_calendar=GetTradingCalendarFromSQL()
        # WriteToMySQL(GetSTFromSQL(trading_calendar,ashare_stock),'ashare_st','winddb')
        ST_data=pd.read_sql(OrderToGetColumnsFromSQL('ann_dt,entry_dt,remove_dt,s_info_windcode,s_type_st','asharest'),connect_mysql)
        ST_data[['ann_dt','entry_dt']]=ST_data[['ann_dt','entry_dt']].applymap(lambda x:pd.to_datetime(x,format='%Y%m%d'))
        ST_data['init_dt']=ST_data[['ann_dt','entry_dt']].min(axis=1)
        ST_data['remove_dt']=ST_data['remove_dt'].map(lambda x:pd.to_datetime(x,format='%Y%m%d') if x is not None else np.nan)
        # remove日期数据中出现了None，设置成样本数据的最后一日
        ST_data['remove_dt'].loc[ST_data['remove_dt'].isnull()]=trading_calendar.iloc[-1]#ST_data[['init_dt','remove_dt']].max().max()
        ST_data['mark']=1.0
        ST_entry=ST_data[['init_dt','s_info_windcode','s_type_st','mark']].set_index(['init_dt','s_info_windcode','s_type_st']).sort_index()
        ST_entry=ST_entry.loc[~ST_entry.index.duplicated(keep='last')].unstack()
        ST_1=ST_entry.iloc[:,0].unstack().reindex(trading_calendar).ffill().fillna(0.0)
        for c in ST_entry.columns[1:]:
            ST_1+=ST_entry[c].unstack().reindex(trading_calendar).ffill().fillna(0.0)
        ST_remove=-ST_data[['remove_dt','s_info_windcode','s_type_st','mark']].set_index(['remove_dt','s_info_windcode','s_type_st']).sort_index()
        ST_remove =ST_remove.loc[~ST_remove.index.duplicated(keep='last')].unstack()
        ST_2 =ST_remove.iloc[:, 0].unstack().reindex(trading_calendar).shift(110).ffill().fillna(0.0)
        for c in ST_remove.columns[1:]:
            ST_2 += ST_remove[c].unstack().reindex(trading_calendar).shift(110).ffill().fillna(0.0)
        # ST的股票标记为1.0
        if ashare_stock is None:
            return ((ST_1+ST_2)>0.0).fillna(0.0)*1.0==0.0
        else:
            return ((ST_1+ST_2)>0.0).reindex(columns=ashare_stock).fillna(0.0)*1.0==0.0 # 类似['000003']的股票只是中途剔除A股，如果一直剔除它，是否会产生look-ahead bias
def GetIPO(limit=110):
    ipo=pd.read_pickle(data_path_local+'IPO')
    return ipo.ffill(limit=limit).fillna(0.0)==0.0

def GetLimit(source='local'):
    if source=='local':
        return pd.read_pickle(data_path_local+'Limit_9.80')
    else:
        ret_d= pd.read_pickle(data_path_local+'adj_prc').pct_change()
        return (1.0*(np.abs(ret_d)>=0.098)).replace(0.0,np.nan).isnull()

def GetIndustryClass():
    return pd.read_pickle(data_path_local+'industry_sec_1st_2nd')
def GetTradingCalendarFromSQL():
    return pd.read_sql('select trade_dt from ashare_tradingcalendar',connect_mysql)['trade_dt']
def GetInvestingDate(freq='m'):
    trading_calendar = GetTradingCalendarFromSQL()
    date_investing = trading_calendar[trading_calendar > '2000']
    if freq.upper()=='M':
        return date_investing.groupby(date_investing.map(lambda x: x.year * 100 + x.month)).first()
    elif freq.upper()=='2W':
        # 感觉有问题，20000104和20000110过于接近
        return date_investing.groupby(pd.DatetimeIndex(date_investing).map(lambda x: x.year * 100+x.month +(x.day>15)*.1 )).first()
    elif freq.upper()=='W':
        return date_investing.groupby(date_investing.map(lambda x: x.year * 100 + x.week)).first()
    else:
        return date_investing

def GetAshareStockCodeFromSQL(history=True):
    if history:
        return pd.read_pickle(data_path_local+'ashare_stock_all')
    else:
        return pd.read_pickle(data_path_local + 'ashare_stock_current')
        #return pd.read_sql('select wind from ashare_stock',connect_mysql)['wind']
def GetFilter(del_finance=True):
    # todo 似乎有问题，2000之后有些截面被删掉只剩1个了
    # todo 剔除涨跌停
    NT=GetNotTradingFromSQL()  # NT 删掉了太多的 2006-07的数据
    ST=GetSTFromSQL()
    IPO=GetIPO(limit=220)
    limit_=GetLimit()
    if del_finance:
        ind=GetIndustryClass()
        return limit_&NT&ST&IPO&(ind!='10')
    else:
        return limit_&NT&ST&IPO

#
def GenerateStockFutureReturns(date_investing, ashare_stock, source='local'):
    if source=='local':
        adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    else:
        adj_prc=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices')
    ret_=adj_prc.shift(1).reindex(index=date_investing,columns=ashare_stock).pct_change().shift(-1)
    return ret_[ret_!=0.0]
def GenerateDailyMarketPortfolioReturns(trading_calendar, ashare_stock, stock_ret_output=False, source='local'):
    # todo 有问题
    #filtered_stock = GetFilter()#.reindex(index=date_investing, columns=ashare_stock).fillna(True)
    if source=='local':
        ret_ = pd.read_pickle(data_path_local + 'adj_prc').pct_change().reindex(columns=ashare_stock)
    else:
        ret_=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices').pct_change().reindex(columns=ashare_stock)
    ret_=ret_[ret_!=0.0]
    market_cap=Size(trading_calendar,ashare_stock)#[filtered_stock]
    if stock_ret_output:
        return (ret_*market_cap).sum(axis=1)/((~ret_.isnull())*market_cap).sum(axis=1),ret_
    else:
        return (ret_*market_cap).sum(axis=1)/((~ret_.isnull())*market_cap).sum(axis=1)
def GetTradingCalendarFromWindDB():
    tc=pd.read_sql(OrderToGetColumnsFromSQL('trade_dt','asharetradingcalendar'),connect_mysql)
    return tc['trade_dt']
trading_calendar = GetTradingCalendarFromSQL()

# 几个常见的中间变量
def MarketCap(tot_market_cap=True,only_fiscal_year_end_without_shift=False,source='local'):
    if source=='local':
        if tot_market_cap: #总市值
            market_cap=pd.read_pickle(data_path_local+'market_cap_total')
        else:
            market_cap = pd.read_pickle(data_path_local + 'market_cap_free')
    else:
        if tot_market_cap: # 总市值
            market_cap=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_mv', 'AShareEODDerivativeIndicator',source='sql')
        else: # 流通市值
            market_cap = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_mv',
                                                            'AShareEODDerivativeIndicator')
    # only_fiscal_year_end=False # 此设定可同时控制所有调用该函数的应用，兼具灵活性和便捷性
    if only_fiscal_year_end_without_shift:
        trade_dt = market_cap.index
        market_cap = GetYearendDataFromDaily(market_cap)
        market_cap.index += DateOffset(months=6)
        market_cap = market_cap.resample('D').ffill().reindex(trade_dt)
    return market_cap

def GetNetOperatingAssetsFromWindDB():
    # https://wenku.baidu.com/view/525a2927dd36a32d737581e9.html
    # FinAssets:货币资金，交易性金融资产，应收利息，应收股利，可供出售金融资产，持有至到期投资
    monetary_cap=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,monetary_cap', 'AShareBalanceSheet',source='local')
    tradable_fin_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tradable_fin_assets', 'AShareBalanceSheet',source='local')
    int_rcv=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,int_rcv', 'AShareBalanceSheet',source='local')
    dvd_rcv=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,dvd_rcv', 'AShareBalanceSheet',source='local')
    fin_assets_avail_for_sale=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,fin_assets_avail_for_sale', 'AShareBalanceSheet',source='local')
    held_to_mty_invest=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,held_to_mty_invest', 'AShareBalanceSheet',source='local')
    # FinLiability: 短期借款，交易性金融负债，应付利息，一年到期非流动负债，长期借款，应付债券
    st_borrow=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,st_borrow', 'AShareBalanceSheet',source='local')
    tradable_fin_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tradable_fin_liab', 'AShareBalanceSheet',source='local')
    int_payable=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,int_payable', 'AShareBalanceSheet',source='local')
    non_cur_liab_due_within_1y=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,non_cur_liab_due_within_1y', 'AShareBalanceSheet',source='local')
    lt_borrow=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,lt_borrow', 'AShareBalanceSheet',source='local')
    bonds_payable=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,bonds_payable', 'AShareBalanceSheet',source='local')
    # equity
    book_value=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
    NOA=st_borrow.fillna(0.0)+tradable_fin_assets.fillna(0.0)+int_payable.fillna(0.0)+non_cur_liab_due_within_1y.fillna(0.0)+lt_borrow.fillna(0.0)+bonds_payable.fillna(0.0)
    -monetary_cap.fillna(0.0)-tradable_fin_assets.fillna(0.0)-int_rcv.fillna(0.0)-dvd_rcv.fillna(0.0)
    -fin_assets_avail_for_sale.fillna(0.0)-held_to_mty_invest.fillna(0.0)+book_value
    return NOA.replace(0.0,np.nan)
def GetEBIT(source='local'):
    # todo 利息 net_int_inc？
    net_inc=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,net_profit_incl_min_int_inc', 'AShareIncome',source=source)
    inc_tax=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inc_tax', 'AShareIncome',source=source)
    net_int_inc=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,net_int_inc', 'AShareIncome',source=source)
    return net_inc+inc_tax.fillna(0.0)-net_int_inc.fillna(0.0)
def GetFamaFrenchFactors(date_investing,ashare_stock,model='FF3'):
    # 问题在于无因子的构造确实麻烦了些，此处仅采用：HML、RMW、CMA都是规模中性，HML自己
    # todo FF5 FF3-china
    #filtered_stock=GetFilter(del_finance=False) # 期初踢掉了太多的股票
    BM=BookToMarket(date_investing,ashare_stock,method='cal')#[filtered_stock]
    size=Size(date_investing,ashare_stock)#[filtered_stock]
    interest_rate_free=GetInterestRateFromLocal(trading_calendar,freq='d')
    ret_=GenerateStockFutureReturns(date_investing,ashare_stock)#[filtered_stock]
    ret = ret.sub(np.exp(interest_rate_free * .01) - 1.0, axis=0)
    qcut_options = {'q': (0.0, 0.3, 0.7,1.0),'q_con':(0.0, 0.5,1.0)};
    qcut_options['labels'] = np.arange(1,len(qcut_options['q']));qcut_options['labels_con'] = np.arange(1,len(qcut_options['q_con']))
    return_BM_size=Returns_BivariatePortfolio(BM.loc['1998':],size.loc['1998':],ret_.loc['1998':],qcut_options,independent=True)
    factor_return=pd.DataFrame(np.nan,index=date_investing,columns=['size','BM'])
    factor_return['BM']=(return_BM_size[qcut_options['labels'][-1]]-return_BM_size[qcut_options['labels'][0]]).groupby(level=0).mean()
    factor_return['size']=return_BM_size.groupby(level=0).apply(lambda x:x.iloc[-1]-x.iloc[0]).mean(axis=1)
    return factor_return
#smb_hml_d=GetFamaFrenchFactors(trading_calendar,ashare_stock)
#factor_return=GetFamaFrenchFactors(trading_calendar,ashare_stock)
#trading_calendar=GetTradingCalendarFromSQL()
#ashare_stock=GetAshareStockCodeFromSQL()
#date_investing=GetInvestingDate(freq='M');date_investing
#factor_return=GetFamaFrenchFactors(date_investing,ashare_stock)
#factor_return
#print(PortfolioAnalysis(factor_return))
#factor_return.to_pickle(data_path_local+'SMB_HML_M')
#tmp=(return_BM_size['5']-return_BM_size['1']).groupby(level=0).mean() # todo 不知道是否是没有清洗数据，导致size和BM都略有高估,但下文的单变量分析似乎又不显著
#tmp=return_BM_size.groupby(level=0).apply(lambda x:x.iloc[-1]-x.iloc[0]).mean(axis=1)
#tmp.mean()/tmp.std()*np.sqrt(len(tmp))
#tmp.mean()/NWest_for_mean(tmp)
#return_=Returns_UnivariatePortfolio(size.reindex(columns=ashare_stock),ret.reindex(columns=ashare_stock),qcut_options)
#tmp=(return_['5']-return_['1']).iloc[:-1]   # todo 单变量分析，size的return太高了
#tmp.mean()/tmp.std()*np.sqrt(len(tmp))
#tmp.mean()/NWest_for_mean(tmp)



## 计算factor或characteristics
def Size(date_investing,ashare_stock,tot_market_cap=True,source='local'):
    # date_investing
    market_cap=MarketCap(tot_market_cap=tot_market_cap,source=source)
    return market_cap.shift(1).reindex(index=date_investing,columns=ashare_stock)
def AssetToMarket(date_investing,ashare_stock):
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    tot_assets=tot_assets.loc[tot_assets.index.month==12]
    tot_assets.index+=DateOffset(months=6)
    # market_cap=GenerateDataframeFromWindDB('trade_dt,s_info_wind_code,s_val_mv', 'AShareEODDerivativeIndicator')
    market_cap=MarketCap(only_fiscal_year_end_without_shift=False)
    # todo 可能需要控制量级
    AM=(tot_assets.reindex(index=market_cap.index,columns=ashare_stock,method='ffill')/market_cap).shift(1).reindex(index=date_investing)
    AM = AM.replace([np.inf, 0.0], np.nan)
    return AM
def BookToMarket(date_investing,ashare_stock,method='cal'):
    if method=='read':
        # todo 检查数据残缺否
        BM= 1.0 / GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_pb_new', 'AShareEODDerivativeIndicator')
    else:
        book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
        book_value=book_value.loc[book_value.index.month==12]
        book_value.index+=DateOffset(months=6)
        #market_cap=GenerateDataframeFromWindDB('trade_dt,s_info_wind_code,s_val_mv', 'AShareEODDerivativeIndicator')
        market_cap=MarketCap(only_fiscal_year_end_without_shift=True)
        BM=book_value.reindex(market_cap.index).ffill()/market_cap
        BM = BM.replace([np.inf, 0.0], np.nan)

    return BM.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def OperatingCashFlowToPrice(date_investing,ashare_stock,method='cal'):
    if method=='read':
        ## todo 是否使用TTM版市现率
        #OCP_ttm= 1.0 / GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_pcf_ocfttm', 'AShareEODDerivativeIndicator')
        OCP= 1.0 / GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_pcf_ocf', 'AShareEODDerivativeIndicator')
    else:
        op_cash_flow=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_cash_flows_oper_act', 'AShareCashFlow',source='local')
        op_cash_flow=op_cash_flow.loc[op_cash_flow.index.month==12]
        op_cash_flow.index+=DateOffset(months=6)
        market_cap=MarketCap(only_fiscal_year_end_without_shift=False)
        # todo 可能需要控制量级
        OCP=op_cash_flow.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap*1e-4
        OCP = OCP.replace([np.inf, 0.0], np.nan)

    return OCP.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def DebtToEquity(date_investing,ashare_stock):
    tot_liab=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_liab', 'asharebalancesheet',source='local')
    tot_liab=tot_liab.loc[tot_liab.index.month==12]
    tot_liab.index+=DateOffset(months=6)
    market_cap=MarketCap(only_fiscal_year_end_without_shift=False)
    DE=tot_liab.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap
    DE = DE.replace([np.inf, 0.0], np.nan)

    return DE.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def LongTermDebtToMarketEquity(date_investing,ashare_stock):
    non_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_non_cur_liab', 'asharebalancesheet',source='local')
    non_cur_liab=non_cur_liab.loc[non_cur_liab.index.month==12]
    non_cur_liab.index+=DateOffset(months=6)
    market_cap=MarketCap(only_fiscal_year_end_without_shift=False)
    LTDME=non_cur_liab.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap
    LTDME = LTDME.replace([np.inf, 0.0], np.nan)

    return LTDME.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def DividendToPrice():
    # todo 似乎数据量不够，所以暂时不需要
    return None
def EarningsToPrice(date_investing,ashare_stock,method='cal'):
    ## todo income before extra: s_fa_deductedprofit?
    if method=='read':
        EP= 1.0 / GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_pe_ttm', 'AShareEODDerivativeIndicator')
    else:
        net_profit=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_profit_incl_min_int_inc', 'AShareIncome',source='local')
        net_profit=net_profit.loc[net_profit.index.month==12]
        net_profit.index+=DateOffset(months=6)
        market_cap=MarketCap(only_fiscal_year_end_without_shift=True)
        # todo 可能需要控制量级
        EP=net_profit.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap
        EP = EP.replace([np.inf, 0.0], np.nan)
    return EP.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def LiabilityGrowth(date_investing,ashare_stock):
    tot_liab=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_liab', 'asharebalancesheet',source='local')
    LG=tot_liab.loc[tot_liab.index.month==12].pct_change()
    LG.index+=DateOffset(months=6)
    LG = LG.replace([np.inf, 0.0], np.nan)

    return LG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def PayoutYield(date_investing,ashare_stock):
    net_profit=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_profit_incl_min_int_inc', 'AShareIncome',source='local')
    net_profit=net_profit.loc[net_profit.index.month==12]
    market_cap=MarketCap(only_fiscal_year_end_without_shift=True)
    book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
    book_value=book_value.loc[book_value.index.month==12]
    tmp=net_profit-book_value
    tmp.index+=DateOffset(months=6)
    PY=tmp.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap
    PY = PY.replace([np.inf, 0.0], np.nan)

    return PY.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def Reversal_60_13(date_investing):
    ## todo 感觉不太需要，删除！
    #adj_prc=GenerateDataframeFromWindDB('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices')
    # 一年按220个交易日计算
    #Revl=(adj_prc.shift(220)-adj_prc.shift(1100))/adj_prc.shift(1100)
    return None# Revl.resample('D').ffill().shift(1).reindex(date_investing).astype(float)
def SustainableGrowth(date_investing,ashare_stock):
    book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
    SG=book_value.loc[book_value.index.month==12].pct_change()
    SG.index+=DateOffset(months=6)
    SG = SG.replace([np.inf, 0.0], np.nan)

    return SG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def SalesGrowthMinusInventoryGrowth(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    SGmIG=oper_rev.loc[oper_rev.index.month==12].pct_change()-inventories.loc[inventories.index.month==12].pct_change().fillna(0.0)
    SGmIG.index+=DateOffset(months=6)
    SGmIG = SGmIG.replace([np.inf, 0.0], np.nan)

    return SGmIG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def SalesToPrice(date_investing,ashare_stock,method='cal'):
    if method=='read':
        SP= 1.0 / GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_ps_ttm', 'AShareEODDerivativeIndicator')
    else:
        oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
        oper_rev=oper_rev.loc[oper_rev.index.month==12]
        oper_rev.index+=DateOffset(months=6)
        market_cap=MarketCap(only_fiscal_year_end_without_shift=True)
        # todo 可能需要控制量级
        SP=oper_rev.reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')/market_cap
        SP = SP.replace([np.inf, 0.0], np.nan)

    return SP.shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def TaxGrowth(date_investing,ashare_stock):
    tax=GenerateDataframeFromWindDBorLocal('report_period,wind_code,inc_tax', 'AShareIncome',source='local')
    tax=tax.loc[tax.index.month==12].pct_change()
    tax.index+=DateOffset(months=6)
    tax = tax.replace([np.inf, 0.0], np.nan)
    return tax.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def Accruals(date_investing,ashare_stock):
    tot_profit=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_profit', 'AShareIncome',source='local')
    op_cash_flow=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_cash_flows_oper_act', 'AShareCashFlow',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    acc=(tot_profit.loc[tot_profit.index.month==12]-op_cash_flow.loc[op_cash_flow.index.month==12].fillna(0.0))\
        /tot_assets.loc[tot_assets.index.month==12]
    acc.index+=DateOffset(months=6)
    acc = acc.replace([np.inf, 0.0], np.nan)
    return acc.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def PercentAccruals(date_investing,ashare_stock):
    tot_profit=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_profit', 'AShareIncome',source='local')
    op_cash_flow=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_cash_flows_oper_act', 'AShareCashFlow',source='local')
    net_profit=GenerateDataframeFromWindDBorLocal('report_period,wind_code,net_profit_incl_min_int_inc', 'AShareIncome',source='local')
    pacc=(tot_profit.loc[tot_profit.index.month==12]-op_cash_flow.loc[op_cash_flow.index.month==12].fillna(0.0))\
         /net_profit.loc[net_profit.index.month==12]
    pacc.index+=DateOffset(months=6)
    pacc = pacc.replace([np.inf, 0.0], np.nan)
    return pacc.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def CapitalExpenditureGrowth(date_investing,ashare_stock):
    capital_exp=GenerateDataframeFromWindDBorLocal('report_period,wind_code,cash_pay_acq_const_fiolta', 'AShareCashFlow',source='local')
    capital_exp=capital_exp.loc[capital_exp.index.month==12].pct_change()
    capital_exp.index+=DateOffset(months=6)
    return capital_exp.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def ChangeInShareholdersEquity(date_investing,ashare_stock):
    book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    dBe=book_value.loc[book_value.index.month==12].diff()/tot_assets.loc[tot_assets.index.month==12].shift(1)
    dBe.index+=DateOffset(months=6)
    dBe = dBe.replace([np.inf, 0.0], np.nan)
    return dBe.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def ChangesInPPEandInventoryToAssets(date_investing,ashare_stock):
    capital_exp=GenerateDataframeFromWindDBorLocal('report_period,wind_code,cash_pay_acq_const_fiolta', 'AShareCashFlow',source='local')
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    dPIA=(capital_exp.loc[capital_exp.index.month==12].diff().fillna(0.0)-inventories.loc[inventories.index.month==12].diff().fillna(0.0))\
         /tot_assets.loc[tot_assets.index.month==12].shift(1)
    dPIA.index+=DateOffset(months=6)
    dPIA=dPIA.replace([np.inf,0.0],np.nan)
    return dPIA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def InvestmentToAssets(date_investing,ashare_stock):
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'asharebalancesheet',source='local')
    IA=tot_assets.loc[tot_assets.index.month==12].pct_change()
    IA.index+=DateOffset(months=6)
    IA = IA.replace([np.inf, 0.0], np.nan)
    return IA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def InventoryChange(date_investing,ashare_stock):
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    IVC=inventories.loc[inventories.index.month==12].diff()/tot_assets.loc[tot_assets.index.month==12].rolling(window=2).mean()
    IVC.index+=DateOffset(months=6)
    IVC = IVC.replace([np.inf, 0.0], np.nan)
    return IVC.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def InventoryGrowth(date_investing,ashare_stock):
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    IG=inventories.loc[inventories.index.month==12].pct_change()
    IG.index+=DateOffset(months=6)
    IG = IG.replace([np.inf, 0.0], np.nan)
    return IG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill')
def NetOperatingAssets(date_investing,ashare_stock):
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_assets', 'asharebalancesheet',source='local')
    net_op_assets=GetNetOperatingAssetsFromWindDB().replace(0.0,np.nan)
    NOA=net_op_assets.loc[net_op_assets.index.month==12]/tot_assets.loc[tot_assets.index.month==12]
    NOA.index+=DateOffset(months=6)
    NOA = NOA.replace([np.inf, 0.0], np.nan)
    return NOA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def AssetTurnover(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    net_op_assets=GetNetOperatingAssetsFromWindDB()
    AT=oper_rev.loc[oper_rev.index.month==12]/net_op_assets.loc[net_op_assets.index.month==12].shift(1)
    AT.index+=DateOffset(months=6)
    AT = AT.replace([np.inf, 0.0], np.nan)
    return AT.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def CashFlowOverAssets():
    return None
def CashProductivity(date_investing,ashare_stock):
    ## （流通市值+长期负债-总资产）/现金及其等价物
    # todo 需要数据对齐，时间+空间
    # todo 数据似乎从2007-07之后才存在足够数据，删除？？？
    #GetYearendDataFromDaily()
    s_dq_mv=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_mv', 'AShareEODDerivativeIndicator')
    s_dq_mv0=GetYearendDataFromDaily(s_dq_mv)
    tot_non_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_non_cur_liab', 'AShareBalanceSheet',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
    cash_cash_equ_end_period=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,cash_cash_equ_end_period', 'AShareCashFlow',source='local')
    CP=(s_dq_mv0+tot_non_cur_liab.loc[tot_non_cur_liab.index.month==12].fillna(0.0)-tot_assets.loc[tot_assets.index.month==12])\
       /cash_cash_equ_end_period.loc[cash_cash_equ_end_period.index.month==12]
    CP.index+=DateOffset(months=6)
    CP = CP.replace([np.inf, 0.0], np.nan)
    return CP.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def CashToAssets(date_investing,ashare_stock):
    # todo 与CashProductivity一样，数据不全，删除？？？
    cash_cash_equ_end_period=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,cash_cash_equ_end_period', 'AShareCashFlow',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
    CA=cash_cash_equ_end_period.loc[cash_cash_equ_end_period.index.month==12]\
       /tot_assets.loc[tot_assets.index.month==12].rolling(window=2).mean()
    CA.index+=DateOffset(months=6)
    CA = CA.replace([np.inf, 0.0], np.nan)
    return CA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def CapitalTurnover(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
    CT=oper_rev.loc[oper_rev.index.month==12]/tot_assets.loc[tot_assets.index.month==12].shift(1)
    CT.index+=DateOffset(months=6)
    CT = CT.replace([np.inf, 0.0], np.nan)
    return CT.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def EBIT2TotRev(date_investing,ashare_stock,method=None):
    # todo 这是啥？？？
    if method=='read':
        ebit2tot_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_ebittogr', 'AShareFinancialIndicator',source='local')
        ebit2tot_rev=ebit2tot_rev.loc[ebit2tot_rev.index.month==12]
    else:
        ebit=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,ebit', 'AShareIncome',source='local')
        tot_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_oper_rev', 'AShareIncome',source='local')
        ebit2tot_rev=ebit.loc[ebit.index.month==12]/tot_rev.loc[tot_rev.index.month==12]
    ebit2tot_rev.index+=DateOffset(months=6)
    ebit2tot_rev = ebit2tot_rev.replace([np.inf, 0.0], np.nan)
    return ebit2tot_rev.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def EarningsYield(date_investing,ashare_stock):
    # EBIT/enterprise_value
    # Enterprise value formula:EV = Market capitalization + Total debt − Cash and cash equivalents
    # todo AshareIncome中EBIT数据太少
    market_cap=MarketCap(only_fiscal_year_end_without_shift=False)# 日频数据
    ebit=GetEBIT(source='local')
    ebit=ebit.loc[ebit.index.month==12].reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')
    tot_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_liab', 'asharebalancesheet',source='local')
    tot_liab=tot_liab.loc[tot_liab.index.month==12].reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')
    cash_cash_equ_end_period=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,cash_cash_equ_end_period', 'AShareCashFlow',source='local')
    cash_cash_equ_end_period=cash_cash_equ_end_period.loc[cash_cash_equ_end_period.index.month==12].reindex(index=market_cap.index,columns=market_cap.columns,method='ffill')
    EY=ebit/(market_cap+tot_liab-cash_cash_equ_end_period.fillna(0.0))
    EY = EY.replace([np.inf, 0.0], np.nan)
    return EY.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def GrossMargins(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    oper_exp=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_exp', 'AShareIncome',source='local')
    oper_rev=oper_rev.loc[oper_rev.index.month==12]
    oper_exp=oper_exp.loc[oper_exp.index.month==12]
    GM=(oper_rev-oper_exp.fillna(0.0))/oper_rev.shift(1)
    GM.index+=DateOffset(months=6)
    GM = GM.replace([np.inf, 0.0], np.nan)
    return GM.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def GrossProfitability(date_investing,ashare_stock):
    # todo 2002-09之后的数据较全
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    oper_exp=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_exp', 'AShareIncome',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
    GP=(oper_rev.loc[oper_rev.index.month==12]-oper_exp.loc[oper_exp.index.month==12].fillna(0.0))/tot_assets.loc[tot_assets.index.month==12].rolling(window=2).mean() # <Firm characteristics and Chinese stocks>中季度算法
    #GP=GP.loc[GP.index.month==12]
    GP.index+=DateOffset(months=6)
    GP = GP.replace([np.inf, 0.0], np.nan)
    return GP.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def NetPayoutOverProfit(date_investing,ashare_stock):
    net_profit_excl_min_int_inc=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,net_profit_excl_min_int_inc', 'AShareIncome',source='local')
    book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
    tot_profit=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_profit', 'AShareIncome',source='local')
    NPOP=(net_profit_excl_min_int_inc.loc[net_profit_excl_min_int_inc.index.month==12]-book_value.loc[book_value.index.month==12].diff())\
         /tot_profit.loc[tot_profit.index.month==12]
    NPOP.index+=DateOffset(months=6)
    NPOP = NPOP.replace([np.inf, 0.0], np.nan)
    return NPOP.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def ReturnOnOperatingAsset(date_investing,ashare_stock):
    s_fa_operateincome=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_operateincome', 'AShareFinancialIndicator',source='local')
    depre=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_stm_is', 'AShareFinancialIndicator',source='local')
    NOP=GetNetOperatingAssetsFromWindDB()
    RNA=(s_fa_operateincome.loc[s_fa_operateincome.index.month==12]-depre.loc[depre.index.month==12].fillna(0))/NOP.loc[NOP.index.month==12].shift(1)
    RNA.index+=DateOffset(months=6)
    RNA= RNA.replace([np.inf, 0.0], np.nan)
    return RNA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def ReturnOnAssets(date_investing,ashare_stock,method='cal'):
    if method.upper()=='READ':
        ROA=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_roa2', 'AShareFinancialIndicator',source='local')
        ROA = ROA.loc[ROA.index.month == 12]
        ROA.index += DateOffset(months=6)
        #ROA=GenerateDataframeFromWindDB('report_period,s_info_windcode,s_fa_roa', 'AShareFinancialIndicator')
    else:
        # todo 与上面的结果差距较大
        tot_profit=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_profit', 'AShareIncome',source='local')
        tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
        ROA=tot_profit.loc[tot_profit.index.month==12]/tot_assets.loc[tot_assets.index.month==12].rolling(window=2).mean()
        ROA.index+=DateOffset(months=6)
        ROA = ROA.replace([np.inf, 0.0], np.nan)
    return ROA.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def ReturnOnEquity(date_investing,ashare_stock,method='read'):
    if method.upper()=='READ':
        ROE=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_roe', 'AShareFinancialIndicator',source='local')
        ROE = ROE.loc[ROE.index.month == 12]
        ROE.index += DateOffset(months=6)
    else:
        net_profit_excl_min_int_inc=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,net_profit_excl_min_int_inc', 'AShareIncome',source='local')
        book_value=GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_shrhldr_eqy_incl_min_int', 'asharebalancesheet',source='local')
        ROE=net_profit_excl_min_int_inc.loc[net_profit_excl_min_int_inc.index.month==12]/book_value.loc[book_value.index.month==12].rolling(window=2).mean()
        ROE.index+=DateOffset(months=6)
        ROE = ROE.replace([np.inf, 0.0], np.nan)
    return ROE.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def TexableIncomeToBookIncome(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    net_profit_excl_min_int_inc=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,net_profit_excl_min_int_inc', 'AShareIncome',source='local')
    TBI=oper_rev.loc[oper_rev.index.month==12]/net_profit_excl_min_int_inc.loc[net_profit_excl_min_int_inc.index.month==12]
    TBI.index+=DateOffset(months=6)
    TBI=TBI.replace([np.inf,0.0],np.nan)
    return TBI.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def ZScore(date_investing,ashare_stock):
    # Dichev(1998):
    # Z= 1.2*(working capital/total assets)+1.4*(retained earnings/total assets)+3.3*(ebit/total assets)
    #   + 0.6*(market cap/total liab)+ (sales/total assets)
    ebit=GetEBIT(source='local')
    s_fa_workingcapital=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_workingcapital', 'AShareFinancialIndicator',source='local')
    tot_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_assets', 'AShareBalanceSheet',source='local')
    s_fa_retainedearnings=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_fa_retainedearnings', 'AShareFinancialIndicator',source='local')
    market_cap=GetYearendDataFromDaily(GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_val_mv', 'AShareEODDerivativeIndicator'))
    tot_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_liab', 'asharebalancesheet',source='local')
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    Z=(1.2*s_fa_workingcapital.loc[s_fa_workingcapital.index.month==12].fillna(0.0)+1.4*s_fa_retainedearnings.loc[s_fa_retainedearnings.index.month==12].fillna(0.0)+3.3*ebit.loc[ebit.index.month==12].fillna(0.0)+oper_rev.loc[oper_rev.index.month==12].fillna(0.0))
    Z=Z.reindex(columns=market_cap.columns)/tot_assets.reindex(columns=market_cap.columns).loc[tot_assets.index.month==12]+0.6*market_cap/tot_liab.reindex(columns=market_cap.columns).loc[tot_liab.index.month==12]
    Z.index+=DateOffset(months=6)
    Z=Z.replace([np.inf,0.0],np.nan)
    return Z.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock,method='ffill').replace(0.0,np.nan)
def ChangeIn6MonthMomentum(date_investing,ashare_stock):
    #adj_prc=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices')
    adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    p0,p1,p2=adj_prc.shift(18),adj_prc.shift(110),adj_prc.shift(220)
    CHMOM=(p0-p1)/p1-(p1-p2)/p2
    return CHMOM.reindex(index=date_investing,columns=ashare_stock,method='ffill').astype(float)
def IndustryMomentum(date_investing,ashare_stock):
    # 以等权重行业收益率作为个股的行业动量
    ind=pd.read_pickle(data_path_local+'industry_sec_1st_2nd') # 03行业使用二级，其他使用一级
    ind_class=set(ind.iloc[-1])
    adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    ret_mom=((adj_prc.shift(1)-adj_prc.shift(20))/adj_prc.shift(20)).reindex(index=date_investing)
    ind_mom=ret_mom.apply(lambda x:x.groupby(ind.loc[x.name]).mean(),axis=1)
    stock_ind_mom=ind.reindex(date_investing).apply(lambda x:x.map(lambda y:ind_mom.loc[x.name,y]),axis=1)
    return stock_ind_mom.reindex(columns=ashare_stock)
def Momentum(date_investing,ashare_stock):
    #adj_prc=GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices')
    adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    mom=(adj_prc.shift(20)-adj_prc.shift(220))/adj_prc.shift(220)
    return mom.reindex(index=date_investing,columns=ashare_stock).astype(float)
def Reversal(date_investing,ashare_stock):
    adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    rev=(adj_prc.shift(1)-adj_prc.shift(20))/adj_prc.shift(20)
    return rev.reindex(index=date_investing,columns=ashare_stock).astype(float)
def Momentum_7_12(date_investing,ashare_stock):
    adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    mom=(adj_prc.shift(110)-adj_prc.shift(220))/adj_prc.shift(220)
    return mom.reindex(index=date_investing,columns=ashare_stock).astype(float)
def BetaDimson(date_investing,ashare_stock,ret_stock,ret_market_portfolio):
    '''
    The Dimson beta, we follow Dimson (1979) to use the lead and the lag of the market return along with the current
        market return to estimate the Dimson beta.
    注意：先运行
        ret_market_portfolio,ret=GenerateMarketPortfolioReturns(freq='D',output_ret_d=True)
    '''
    #ret_stock=ret_stock.fillna(0.0)
    beta_dimson=pd.DataFrame(np.nan,index=date_investing,columns=ret_stock.columns)
    X=pd.concat([ret_market_portfolio.shift(2),ret_market_portfolio.shift(1),ret_market_portfolio,
                 ret_market_portfolio.shift(-1),ret_market_portfolio.shift(-2)],axis=1)#.fillna(0.0)
    X=sm.add_constant(X)
    for t in date_investing:
        try:
            #tmpX=X.loc[t-DateOffset(months=2):t-Day()].iloc[:-2].apply(lambda x:x-x.mean())
            #tmpY=ret_stock.loc[tmpX.index].apply(lambda x:x-x.mean()).fillna(0.0)
            #beta_dimson.loc[t]=(np.linalg.pinv(tmpX.values.T@tmpX.values)@tmpX.values@tmpY).sum()
            beta_dimson.loc[t]=ret_stock.loc[t-Day(45):t-Day()].apply(lambda Y:sm.OLS(Y,X.loc[Y.index]).fit().params[1:].sum())
        except:
            continue
    beta_dimson=beta_dimson.astype(float)
    return beta_dimson[(beta_dimson!=0) & (beta_dimson!=np.inf)].reindex(columns=ashare_stock)
def BetaDownside(date_investing,ashare_stock,ret_stock,ret_market_portfolio):
    '''
    注意：先运行
        ret_market_portfolio,ret=GenerateMarketPortfolioReturns(freq='D',output_ret_d=True)
    '''
    beta_downside=pd.DataFrame(np.nan,index=date_investing,columns=ret_stock.columns)
    for t in date_investing:
        #try:
        tmpX=ret_market_portfolio.loc[t-DateOffset(months=3):t-Day()].agg(lambda x:x[x>x.mean()])
        beta_downside.loc[t]=ret_stock.loc[tmpX.index].corrwith(tmpX)*ret_stock.loc[tmpX.index].std()/tmpX.std()
        #except:
        #    continue
    #beta_downside=beta_downside.astype(float)
    return beta_downside[beta_downside!=0.0].reindex(columns=ashare_stock)
def BetaMarket(date_investing,ashare_stock,ret_stock,ret_market_portfolio):
    '''
    注意：先运行
        ret_market_portfolio,ret=GenerateMarketPortfolioReturns(freq='D',output_ret_d=True)
    '''
    beta_market=pd.DataFrame(np.nan,index=date_investing,columns=ret_stock.columns)
    for t in date_investing:
        #try:
        #tmpX=ret_market_portfolio.loc[t-DateOffset(months=2):t-Day()]
        #tmpY=ret_stock.loc[t-DateOffset(months=2):t-Day()]
        beta_market.loc[t]=ret_stock.loc[t-DateOffset(months=2):t-Day()].corrwith(ret_market_portfolio.loc[t-DateOffset(months=2):t-Day()])*ret_stock.loc[t-DateOffset(months=2):t-Day()].std()/ret_market_portfolio.loc[t-DateOffset(months=2):t-Day()].std()
        #except:
        #    continue
    #beta_market=beta_market.astype(float)
    return beta_market[beta_market!=0.0].reindex(columns=ashare_stock)
def BetaSquared(date_investing,ashare_stock,ret_stock,ret_market_portfolio):
    '''
    注意：先运行
        ret_market_portfolio,ret=GenerateMarketPortfolioReturns(freq='D',output_ret_d=True)
    '''
    return BetaMarket(date_investing,ashare_stock,ret_stock,ret_market_portfolio)**2.0
def BetaFP(date_investing,ashare_stock,ret_stock,ret_market_portfolio):
    '''
    Frazzini adn Pedersen(2014)
    use one-day log returns to estimate volatilities and
        overlapping three-day log returns for correlation to control for nonsynchronous trading (which affects only correlations).
    :return:
    '''
    #adj_prc = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_adjclose', 'AShareEODPrices')
    #adj_prc=pd.read_pickle(data_path_local+'adj_prc')
    #ret_market,ret=GenerateMarketPortfolioReturns(trading_calendar[trading_calendar>'2000'],ashare_stock,stock_ret_output=True)
    #ret_3d=(1.0+ret_stock).rolling(window=3).apply(lambda x:x.cumprod()[-1],raw=True).shift(-2)-1.0
    ret_3d=ret_stock.rolling(window=3).sum().shift(-2)
    ret_market_3d=ret_market_portfolio.rolling(window=3).sum().shift(-2)
    beta_FP=pd.DataFrame(np.nan,index=date_investing,columns=ret_stock.columns)
    for t in date_investing:
        #try:
        beta_FP.loc[t]=\
            ret_3d.loc[t-DateOffset(months=2):t-Day(3)].corrwith(ret_market_3d.loc[t-DateOffset(months=2):t-Day(3)])\
                       *ret_stock.loc[t-DateOffset(months=2):t-Day()].std()/ret_market_portfolio.loc[t-DateOffset(months=2):t-Day()].std()
        #beta_FP.apply(lambda x:ret_3d.loc[x.name - DateOffset(months=2):x.name - Day(3)].corrwith(ret_market_3d.loc[x.name - DateOffset(months=2):x.name - Day(3)]) \
        #* ret_stock.loc[x.name - DateOffset(months=2):x.name - Day()].std() / ret_market_portfolio.loc[
        #                                                                      x.name - DateOffset(months=2):x.name - Day()].std(),axis=1)

        #except:
        #    continue
    return beta_FP.reindex(columns=ashare_stock)
def IdiosyncraticVolatility(date_investing,ashare_stock,ret_stock,ret_market_portfolio,method='FF3'):
    ivol = pd.DataFrame(np.nan, index=date_investing, columns=ashare_stock)
    if method=='FF3':
        smb_hml=pd.read_pickle(data_path_local+'SMB_HML_D')
    #tc=GetTradingCalendarFromSQL()
    #market_port_ret,ret=GenerateMarketPortfolioReturns(tc[tc>'2000'],ashare_stock,stock_ret_output=True,source='local')
    #ret=pd.read_pickle(data_path_local+'adj_prc').pct_change()
        X=pd.concat([ret_market_portfolio,smb_hml],axis=1)
    #tmp=ivol.apply(lambda x:x.map(lambda y:sm.OLS(ret_stock.loc[x.name-Day(45):x.name-Day()],X.loc[x.name-Day(45):x.name-Day()]).fit().resid.std()),axis=1)
        for t in ivol.index:
        #try:
            # 不要使用sm.OLS,慢
        #ivol.loc[t]=ret_stock_tmp.loc[t-DateOffset(months=1):t-Day()].apply(lambda y:sm.OLS(y,X.loc[y.index].values).fit().resid.std())
            ivol.loc[t]=GenerateErrorFromOLSbyMatrix(ret_stock.loc[t-Day(45):t-Day(15)],X.loc[t-Day(45):t-Day(15)]).std()
        #except:
        #    continue
    else:
        for t in ivol.index:
            # todo
            ivol.loc[t]=\
                GenerateErrorFromOLSbyMatrix(ret_stock.loc[t-Day(45):t-Day(15)],ret_market_portfolio.loc[t-Day(45):t-Day(15)]).std()
    return ivol[ivol>0.0].reindex(columns=ashare_stock)
def Illiquidity(date_investing,ashare_stock):
    # abs(ret)/trade_amount
    #limit_=GetLimit()
    ret_d= pd.read_pickle(data_path_local+'adj_prc').pct_change()#[limit_]
    s_dq_amount = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
    s_dq_amount=s_dq_amount[s_dq_amount>0.0]
    illiq_d=np.abs(ret_d)/s_dq_amount*1e9
    #illiq=pd.DataFrame(np.nan,index=date_investing,columns=ashare_stock)
    illiq=illiq_d.reindex(date_investing).apply(lambda x:illiq_d.loc[x.name-Day(60):x.name-Day(3)].mean(),axis=1)
    #for t in date_investing:
    #    try:
    #        illiq.loc[t]=illiq_d.loc[t-DateOffset(months=1):t-Day()].mean()
    #    except:
    #        continue
    return illiq.reindex(columns=ashare_stock)
def MaxDailyReturn(date_investing,ashare_stock):
    # 添加筛选
    ret_= pd.read_pickle(data_path_local+'adj_prc').pct_change()
    ret_=ret_[ret_!=0]
    MDR = pd.DataFrame(np.nan, index=date_investing, columns=ashare_stock)
    for t in date_investing:
        try:
            MDR.loc[t]=ret_.loc[t-DateOffset(months=1):t-Day()].max()
        except:
            continue
    return MDR

def Price():
    # todo ???
    return None
def TradingAmount(date_investing,ashare_stock):
    # 过去一段时间的log(trading_volume)
    s_dq_amount = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
    s_dq_amount[s_dq_amount<=0.0]=np.nan
    Rvol=s_dq_amount.loc[date_investing].apply(lambda x:np.log(s_dq_amount.loc[x.name-Day(30):x.name-Day()].mean()),axis=1)
    return Rvol.reindex(columns=ashare_stock)
def VolatilityOfTradingAmount(date_investing,ashare_stock):
    s_dq_amount = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
    log_amount=np.log(s_dq_amount[s_dq_amount>0.0])
    VTA = pd.DataFrame(np.nan, index=date_investing, columns=ashare_stock)
    for t in date_investing:
        try:
            VTA.loc[t] = log_amount.loc[t - DateOffset(months=1):t - Day()].std()
        except:
            continue
    return VTA.astype(float)
def VolatilityOfTurnover(date_investing,ashare_stock):
    s_dq_mv = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_mv', 'AShareEODDerivativeIndicator')
    s_dq_amount = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
    s_dq_mv=s_dq_mv[s_dq_mv>0.0]
    s_dq_amount = s_dq_amount[s_dq_amount > 0.0]
    TO_d = s_dq_amount / s_dq_mv
    VTO = pd.DataFrame(np.nan, index=date_investing, columns=ashare_stock)
    for t in date_investing:
        try:
            VTO.loc[t] = TO_d.loc[t - DateOffset(months=1):t - Day()].std()
        except:
            continue
    return VTO.astype(float)
def VolatilityOfReturns(date_investing,ashare_stock):
    ret_ = pd.read_pickle(data_path_local+'adj_prc').pct_change()
    ret_ = ret_[ret_!=0.0]
    VoR=pd.DataFrame(np.nan,index=date_investing,columns=ashare_stock)
    for t in date_investing:
        #try:
        VoR.loc[t]=ret_.loc[t-DateOffset(months=2):t-Day()].std()
        #except:
        #    continue
    return VoR.astype(float)
def Turnover(date_investing,ashare_stock):
    s_dq_mv = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_mv', 'AShareEODDerivativeIndicator')
    s_dq_amount = GenerateDataframeFromWindDBorLocal('trade_dt,s_info_windcode,s_dq_amount', 'AShareEODPrices')
    s_dq_mv = s_dq_mv[s_dq_mv > 0.0]
    s_dq_amount = s_dq_amount[s_dq_amount > 0.0]
    TO_d=s_dq_amount/s_dq_mv
    TO=pd.DataFrame(np.nan,index=date_investing,columns=ashare_stock)
    for t in date_investing:
        try:
            TO.loc[t]=TO_d.loc[t-DateOffset(months=1):t-Day()].mean()
        except:
            continue
    return TO.astype(float)
def CashflowToDebt(date_investing,ashare_stock):
    ebit=GetEBIT()
    s_stm_is=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,s_stm_is', 'AShareFinancialIndicator',source='local')
    ebitda=ebit.loc[ebit.index.month==12]+s_stm_is.loc[s_stm_is.index.month==12].fillna(0.0)
    tot_liab = GenerateDataframeFromWindDBorLocal('report_period,wind_code,tot_liab', 'asharebalancesheet',source='local')
    CFD=ebitda/tot_liab.loc[tot_liab.index.month==12].rolling(window=2).mean()
    CFD.index+=DateOffset(months=6)
    CFD=CFD.replace([np.inf,0.0],np.nan)
    return CFD.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def CurrentRatio(date_investing,ashare_stock):
    # 流动资产/流动负债
    # todo 2005年之后金融行业的流动比率似乎不存在
    tot_cur_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_assets', 'asharebalancesheet',source='local')
    tot_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_liab', 'asharebalancesheet',source='local')
    current=tot_cur_assets[tot_cur_assets.index.month==12]/tot_cur_liab[tot_cur_liab.index.month==12]
    current.index += DateOffset(months=5)
    current=current.replace([np.inf,0.0],np.nan)
    return current.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock)
def CurrentRatioGrowth(date_investing,ashare_stock):
    tot_cur_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_assets', 'asharebalancesheet',source='local')
    tot_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_liab', 'asharebalancesheet',source='local')
    CRG=(tot_cur_assets[tot_cur_assets.index.month==12]/tot_cur_liab[tot_cur_liab.index.month==12]).pct_change()
    CRG.index += DateOffset(months=6)
    CRG=CRG.replace([np.inf,0.0],np.nan)
    return CRG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock)
def QuickRatio(date_investing,ashare_stock):
    tot_cur_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_assets', 'asharebalancesheet',source='local')
    tot_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_liab', 'asharebalancesheet',source='local')
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    QR=(tot_cur_assets[tot_cur_assets.index.month==12]-inventories.loc[inventories.index.month==12].fillna(0.0))/tot_cur_liab[tot_cur_liab.index.month==12]
    QR.index+=DateOffset(months=6)
    QR=QR.replace([np.inf,0.0],np.nan)
    return QR.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock)
def QuickRatioGrowth(date_investing,ashare_stock):
    tot_cur_assets=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_assets', 'asharebalancesheet',source='local')
    tot_cur_liab=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,tot_cur_liab', 'asharebalancesheet',source='local')
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    QRG=((tot_cur_assets[tot_cur_assets.index.month==12]-inventories.loc[inventories.index.month==12].fillna(0.0))/tot_cur_liab[tot_cur_liab.index.month==12]).pct_change()
    QRG.index+=DateOffset(months=6)
    QRG=QRG.replace([np.inf,0.0],np.nan)
    return QRG.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock)
def SalesToCash(date_investing,ashare_stock):
    # todo 删除
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    cash_cash_equ_end_period=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,cash_cash_equ_end_period', 'AShareCashFlow',source='local')
    SC=oper_rev.loc[oper_rev.index.month==12]/cash_cash_equ_end_period.loc[cash_cash_equ_end_period.index.month==12]
    SC.index+=DateOffset(months=6)
    return SC.resample('D').ffill().shift(1).reindex(index=date_investing,columns=ashare_stock).astype(float)
def SalesToInventory(date_investing,ashare_stock):
    oper_rev=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,oper_rev', 'AShareIncome',source='local')
    inventories=GenerateDataframeFromWindDBorLocal('report_period,s_info_windcode,inventories', 'AShareBalanceSheet',source='local')
    SI=(oper_rev.loc[oper_rev.index.month==12]/inventories.loc[inventories.index.month==12]).replace(np.inf,np.nan)
    SI.index+=DateOffset(months=6)
    SI=SI[SI>0.0]
    return SI.reindex(index=trading_calendar,method='ffill').shift(1).reindex(index=date_investing,columns=ashare_stock)

#def TestErr(df):
#    return (df==np.inf).sum().sum(),(df==0.0).sum().sum(),(df<0.0).sum().sum()










