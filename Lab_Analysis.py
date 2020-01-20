# 目标：怎样做才能构建SR最大（尽可能）的组合（样本外）
# todo 非有效因子是否有用？（与只有有效因子的情况做对比）
# todo 样本外PCA是否有效？ 与1/n策略比较？
# todo 样本外Sparse PCA是否有效？
# todo 行业中性化，降低股票之间的comovement
# todo Sparse PCA方法不依赖于效用函数假设（疑问：PPP对风险厌恶参数敏感吗？？？老外的验证结果显示gamma<=5时，似乎SR差距不是很显著）
# todo 考虑卖空约束和交易成本时(交易成本策略可设置为与PPP相同)，Sparse PCA是否优于PPP

import pandas as pd
from FactorZooUsingSQL import *
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.decomposition import SparsePCA
from scipy import stats
from scipy.optimize import minimize,Bounds,LinearConstraint,NonlinearConstraint,BFGS,SR1

warnings.filterwarnings('ignore')

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
options_mysql = {'server': 'localhost',
                     'user': 'root',
                     'password': '1234567890',
                     'database': 'winddb'}
connect_mysql = ConnectMySQL(**options_mysql)
data_path_local = 'E:/data/winddb/' # '/Users/harbes/PycharmProjects/data/winddb/' #
signal_names = '''AM,BM,CFP,DER,DLME,EP,LG,PY,SG,SMI,SP,TG,ACC,PACC,CAPXG,dBe,dPIA,IA,IVC,IVG,NOA,ATO,CTO,EY,GM,GP,NPOP,RNA,ROA,ROE,TBI,Z_score,CHMOM,INDMOM,REV,MOM12M,MOM6M,B_DIM,B_DN,BETA,BETASQ,B_FP,IVOL,ILLIQ,MAXRET,RVOL,SIZE,STD_RVOL,STD_TURN,RETVOL,TURN,CFD,CR,CRG,QR,QRG,SI'''

def GenerateMarketPortfolioReturnsAccordingInvestingDate(ret,w=None):
    if w is None:
        return ret.mean(axis=1),(~ret.isnull()).agg(lambda x:x/x.sum(),axis=1)
    else:
        return (ret*w).sum(axis=1)/((~ret.isnull())*w).sum(axis=1),((~ret.isnull())*w).agg(lambda x:x/x.sum(),axis=1)
def GenerateStockWeights(theta,charas,w_bench=None):
    if w_bench is None:
        return charas.fillna(0.0).mul(theta, axis=1).sum(axis=1).unstack()
    else:
        return charas.fillna(0.0).mul(theta, axis=1).sum(axis=1).add(w_bench.stack().reindex(charas.index)).unstack()


#trading_calendar = GetTradingCalendarFromSQL()
trading_calendar=pd.read_pickle(data_path_local+'trading_calendar')
#ashare_stock = GetAshareStockCodeFromSQL()
ashare_stock=pd.read_pickle(data_path_local+'ashare_stock_all')

freq_list = ['M', '2W', 'W']
freq=freq_list[0]

#date_investing = GetInvestingDate(freq=freq)
date_investing=pd.read_pickle(data_path_local+'date_investing_'+freq)
interest_rate_free=GetInterestRateFromLocal(date_investing,freq=freq)
interest_rate_free_d=GetInterestRateFromLocal(trading_calendar,freq='d')
filtered_stock=GetFilter(del_finance=False).reindex(date_investing)
ret = GenerateStockFutureReturns(date_investing,ashare_stock).sub(np.exp(interest_rate_free*.01)-1.0,axis=0) # [filtered_stock] # 每个investing date对应的收益率r_{t+1}
#ret = ret.sub(np.exp(interest_rate_free*.01)-1.0,axis=0)
SIZE = Size(date_investing, ashare_stock)  # [filtered_stock] # EW:-3.415907 ; VW: -2.863930

predictors_all=pd.read_pickle(data_path_local + 'predictors_all_' + freq).replace(np.inf,np.nan)
ret_stock_d=pd.read_pickle(data_path_local+'adj_prc').pct_change().replace([np.inf,0.0],np.nan)
ret_stock_d=ret_stock_d.sub(np.exp(interest_rate_free_d*.01)-1.0,axis=0)
#ret_benchmark,_=GenerateMarketPortfolioReturnsAccordingInvestingDate(ret,size=SIZE)
# 计算 特征组合的收益
# 检查数据维度，
#ret.size # 926640
#ret.stack().size # 465162
#len(predictors_all) # 483701

# 特征标准化
# winsorize处理
chara_port_standard=predictors_all.groupby(level=0).apply(lambda x:x.apply(lambda y:stats.mstats.winsorize(y,limits=[0.01,0.01]),axis=0)).groupby(level=0).apply(lambda x:(x-x.mean())/x.std())
#chara_port_standard=predictors_all.groupby(level=0).apply(lambda x:(x-x.mean())/x.std())

# 检查是否标准化
#chara_port_standard.groupby(level=0).agg(['mean','std'])
# 特征组合收益率
#ret_chara_port=chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).sum();
# 特征与截面ret之间的协方差【由于标准化，所以cov=rho=beta(ret~chara)；特征与截面ret之间的相关系数=cov(chara,ret)/disp(chara)/disp(ret)】
cov_chara_ret=100.0*chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).mean();
# beta(chara~ret)
#beta_chara_ret=chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).mean()/ret.std(axis=1)

#(ret_chara_port.std(axis=1)/ret_chara_port.std(axis=1).mean()).plot(label='ret_chara_port_disp');
#(cov_chara_ret.std(axis=1)/cov_chara_ret.std(axis=1).mean()).plot(label='cov_disp');
#(ret.std(axis=1)/ret.std(axis=1).mean()).plot(label='ret_disp')
#plt.legend()
#plt.show()

#cov_chara_ret.std(axis=1).idxmax()
#cov_chara_ret.loc[cov_chara_ret.std(axis=1).idxmax()]
# 不同时期最有效的预测变量
#cov_chara_ret.agg(['idxmin','idxmax'],axis=1)

# todo 组合分析检验因子
#qcut_options = {'q': np.arange(0, 1.00001, 0.2), 'q_con': np.arange(0, 1.00001, 0.2)}
qcut_options = {'q': (0.0, 0.3, 0.7, 1.0), 'q_con': (0.0, 0.5, 1.0)};
qcut_options['labels'] = np.arange(1, len(qcut_options['q']));
qcut_options['labels_con'] = np.arange(1, len(qcut_options['q_con']))
factor_test=pd.DataFrame(np.nan,index=['mean','t_NW','alpha','t_alpha_NW'],columns=chara_port_standard.columns)
port_market,_=GenerateMarketPortfolioReturnsAccordingInvestingDate(ret,w=SIZE)
port_market=port_market.sub(np.exp(interest_rate_free*.01)-1.0,axis=0)
smb_hml=pd.read_pickle(data_path_local+'SMB_HML_'+freq.upper())
X=pd.concat([port_market,smb_hml],axis=1).loc[:'2019-11']*100.0
XXX=np.linalg.inv(X.agg(lambda x:x-x.mean()).T@X.agg(lambda x:x-x.mean()))@X.agg(lambda x:x-x.mean()).T#@port_long_short.loc[:'2019-11'].fillna(port_long_short.mean())
def GenerateAlphaAndErrorByOLS(Y,X,XXX):
    beta=XXX@Y.agg(lambda x:x-x.mean()).fillna(0.0)
    ee=Y-X.values@beta
    return ee
for c in chara_port_standard.columns:
    port_long_short=Returns_UnivariatePortfolio(chara_port_standard[c].unstack().reindex(index=ret.index,columns=ret.columns),ret*100.0,qcut_options,weights=SIZE).apply(lambda x:x.iloc[-1]-x.iloc[0],axis=1)
    # 双变量组合分析总是出现问题：1、filtered_stock 筛选后的股票太少;2、输出结果总是nan
    #port_long_short=Returns_BivariatePortfolio(chara_port_standard[c].unstack().reindex(index=ret.index,columns=ret.columns),SIZE,ret*100.0,qcut_options,weights=SIZE,independent=False).apply(lambda x:x.iloc[-1]-x.iloc[0],axis=1)
    factor_test.loc[(['mean','t_NW'],c)]=\
        PortfolioAnalysis(port_long_short).loc[(['mean','t_NW_adjusted'],slice(None))][0].values
    e=GenerateAlphaAndErrorByOLS(port_long_short.reindex(date_investing.iloc[:-1]),X,XXX)
    factor_test.loc[('alpha', c)]= e.mean()
    factor_test.loc[('t_alpha_NW', c)]=\
        factor_test.loc[('alpha',c)]/NWest(e.fillna(e.mean()),X)


factor_test.to_pickle(data_path_local+'Factor_test_VW')
factor_test.to_csv(data_path_local+'Factor_test_VW.csv')





# todo 直接对57个特征组合使用1/n策略
# todo 尚未考虑交易成本、卖空约束
PortfolioAnalysis(cov_chara_ret.T.mul(np.sign(cov_chara_ret.mean()),axis=0).T.mean(axis=1)) # 效果很好（SR=0.72;t=9.1）
'''
mean           0.150524
std            0.239702
SR             0.627965
SR_ann         1.451926
t              9.708098
t_NW_adjusted  7.861404
'''
cov_chara_ret['BM'].plot();plt.show()
cov_chara_ret.apply(lambda x:x.autocorr(lag=4))



#todo  PCA using numpy
# todo 如何选择reconstructed portfolio？ Sharpe ratio？
# PCA方法生成的权重矩阵有个问题：权重和不为0（平方和为0，因为单位向量）
s,V_T=np.linalg.eigh(cov_chara_ret.cov())
#(cov_chara_ret.fillna(0.0)@V_T[:,:15]).replace(0.0,np.nan).plot();plt.show()
ret_port_reconstrcut=(cov_chara_ret.fillna(0.0)@V_T[:,:]).replace(0.0,np.nan)
# 调整符号
for i in range(V_T.shape[1]):
    V_T[:,i]=V_T[:,i]*np.sign(ret_port_reconstrcut[i].mean())
ret_port_reconstrcut = (cov_chara_ret.fillna(0.0) @ V_T[:, :]).replace(0.0, np.nan)

select1=ret_port_reconstrcut.columns[PortfolioAnalysis(ret_port_reconstrcut).loc['t_NW_adjusted']>2.0]
len(select1) # 25个
# 历史表现最优的几个组合
select2=np.abs(PortfolioAnalysis(ret_port_reconstrcut).loc['SR']).sort_values(ascending=False).index[:30]
PortfolioAnalysis(ret_port_reconstrcut[select2])
# 历史最优组合；不同特征上赋予的权重，权重成梯队状，且一定范围内相近---》1/n策略
pd.DataFrame(np.abs(V_T[:,select2[0]]),index=signal_names.split(','),columns=['w']).sort_values(by=['w'],ascending=False)
PortfolioAnalysis(ret_port_reconstrcut[select2[0]])
ret_port_reconstrcut[select2[0]].plot();plt.show()
ret_port_reconstrcut[select2[0]].rolling(window=24).std().plot();plt.show()
# 权重分布
#weights_select=np.abs(pd.DataFrame(V_T[:,select1],index=signal_names.split(','),columns=select1)).sort_values(by=select2[0],ascending=False)
weights_description=\
    pd.DataFrame(V_T[:,select1],index=signal_names.split(','),columns=select1).sort_values(by=select2[0],ascending=False).agg(['min','max','mean','std'],axis=1)#.sort_values(ascending=False)

# t>2.0 的组合采取 1/N 策略，样本内效果非常好
PortfolioAnalysis(ret_port_reconstrcut[select1].mean(axis=1))
'''
mean            0.273983
std             0.336963
SR              0.813094
SR_ann          1.724199
t              12.570125
t_NW_adjusted  10.028969
'''



# todo PCA的样本外效果;结果与样本外观测期的选择、Day()的参数、cov还是corr有关
def WeightsEstimatedFromPCA(ret_p,n_com=15):
    # 删除还是填充
    #ret_port=ret_p.reindex(columns=ret_p.columns[ret_p.isnull().mean()<0.5])
    ret_port=ret_p.apply(lambda x:x.fillna(x.mean())).dropna(how='all',axis=1)
    s,V=np.linalg.eigh(ret_port.cov())
    ret_port_reconstrcut = (ret_port.fillna(0.0) @ V[:, :]).replace(0.0, np.nan)
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] * np.sign(ret_port_reconstrcut[i].mean())
    ret_port_reconstrcut = (ret_port.fillna(0.0) @ V[:, :]).replace(0.0, np.nan)
    PortfolioAnalysis(ret_port_reconstrcut)
    # 根据组合的组合的平均收益，调整组合的符号
    weights = pd.DataFrame(V, index=ret_port.columns);
    # 按t值选，还是按SR选择[实证结果显示按t值排序效果更好]
    select_port=np.abs(PortfolioAnalysis(ret_port_reconstrcut)).T.sort_values(by='t_NW_adjusted',ascending=False).index[:n_com]
    return weights[select_port]#.apply(lambda x:x/np.abs(x).sum())
n_com=15
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
#ret_top_ports=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=range(1))
w=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product([t_range_out_of_sample,signal_names.split(',')]),columns=range(n_com))
for t in t_range_out_of_sample:
    w.loc[t]=WeightsEstimatedFromPCA(cov_chara_ret.loc[:t-Day()],n_com).reindex(index=cov_chara_ret.columns).values
ret_optimal_port=w.mul(cov_chara_ret.loc[t_range_out_of_sample].stack().reindex(w.index),axis=0).groupby(level=0).sum()
PortfolioAnalysis(ret_optimal_port).loc['t_NW_adjusted'].sort_values()
# 1/n 策略，样本外效果非常好？？？
#PortfolioAnalysis((cov_chara_ret.loc[t_range_out_of_sample]*w.mean(axis=1).unstack()).sum(axis=1))
PortfolioAnalysis(ret_optimal_port.mean(axis=1))
chara_port_standard
w.mean(axis=1).unstack()

'''
mean            0.229686
std             0.276664
SR              0.830199
t              11.107300
t_NW_adjusted  10.081646
'''
w.mean(axis=1).unstack()


# todo Sparse PCA 样本内结果似乎没有PPP效果好
n_com=20
transformer = SparsePCA(n_components=n_com,alpha=0.6)#, random_state=0)
# 输入原始矩阵（＋标准化）还是输入协方差矩阵？？？
# 注意输入的变量量级不要太小,也不要太大，100比较适合（未/std）
transformer.fit(cov_chara_ret.dropna(how='all',axis=0).agg(lambda x:x-x.mean()).fillna(0.0)*100.0)#.cov())
#transformer.transform(cov_chara_ret.dropna(how='all',axis=0).fillna(0.0))#.apply(lambda x:x.where(~x.isnull(),x.mean()),axis=0))#,index=date_investing[date_investing<'2019-12'])
transformer.transform(cov_chara_ret.dropna(how='all',axis=0).agg(lambda x:x-x.mean()).fillna(0.0)*100.0)#.apply(lambda x:x.where(~x.isnull(),x.mean()),axis=0))#,index=date_investing[date_investing<'2019-12'])
# 根据组合的组合的平均收益，调整组合的符号
weights=pd.DataFrame(transformer.components_,columns=signal_names.split(',')).T;weights
ret_transformed_port=(cov_chara_ret.fillna(0.0)@transformer.components_.T).replace(0.0,np.nan);ret_transformed_port
for c in weights.columns:
    # 调整权重的符号，并且scale权重(对于)
    weights[c]=weights[c]*np.sign(ret_transformed_port[c].mean())/np.abs(weights[c]).sum()
ret_transformed_port=(cov_chara_ret.fillna(0.0)@weights).replace(0.0,np.nan);#ret_transformed_port
res=PortfolioAnalysis(ret_transformed_port.dropna(how='all',axis=1)).T.sort_values(by='SR',ascending=False);res
weights[res.index[:1]].replace(0.0,np.nan) # 最优组合总是配置在与交易成本(换手率、交易量、特质性波动率、最大收益率)相关的变量中

# 1/n策略效果更好
res_m=PortfolioAnalysis(ret_transformed_port[res.index[:int(n_com*.67)]].mean(axis=1));res_m # 平均策略
weights[res.index[:int(n_com*.67)]].mean(axis=1)#.replace(0.0,np.nan) # 最优组合总是配置在与交易成本(换手率、交易量、特质性波动率、最大收益率)相关的变量中
w=GenerateStockWeights(weights[res.index[:int(n_com*.67)]].mean(axis=1),chara_port_standard,weights_bench)


(w<0.0).loc[:'2019-11'].mean(axis=1).plot();plt.show() # 空头需求越来越高



# todo Sparse PCA 样本外表现
def WeightsEstimatedFromSparsePCA(ret_port,n_com=25):
    tf = SparsePCA(n_components=n_com)  # , random_state=0)
    tf.fit(ret_port.agg(lambda x:x-x.mean()).fillna(0.0))  # 注意量级
    tf.transform(ret_port.fillna(0.0)) # .apply(lambda x:x.where(~x.isnull(),x.mean()),axis=0))#,index=date_investing[date_investing<'2019-12'])
    # 根据组合的组合的平均收益，调整组合的符号
    weights = pd.DataFrame(tf.components_, columns=signal_names.split(',')).T;
    ret_transformed_port = (cov_chara_ret.fillna(0.0) @ weights).replace(0.0, np.nan);
    for c in weights.columns:
        weights[c] = weights[c] * np.sign(ret_transformed_port[c].mean()) / np.abs(weights[c]).sum()
    ret_transformed_port = (cov_chara_ret.fillna(0.0) @ weights).replace(0.0, np.nan);
    # 按t值选，还是按SR选择
    select_port=np.abs(PortfolioAnalysis(ret_transformed_port.dropna(how='all', axis=1))).T.sort_values(by='SR',ascending=False).index[:int(n_com*0.67)]
    for p in select_port:
        weights[p]*=np.sign(ret_transformed_port[p].mean())
    return weights[select_port]
n_com=25
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
ret_optimal_port=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=range(1))
# w的维度(T*A)*n_com
w=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product([t_range_out_of_sample,signal_names.split(',')]),columns=range(int(n_com*0.67)))
for t in t_range_out_of_sample:
    w.loc[t]=WeightsEstimatedFromSparsePCA(100.0*cov_chara_ret.loc[t-DateOffset(months=120):t-Day()],n_com).values
ret_optimal_port=w.mul(cov_chara_ret.loc[t_range_out_of_sample].stack().reindex(w.index),axis=0).groupby(level=0).sum()
PortfolioAnalysis(ret_optimal_port.mean(axis=1)).T.sort_values(by='SR',ascending=False) # 平均策略的样本外效果也很好，SR=0.719
# 1/n 策略
ret_1_n=w.loc[:,:11].mean(axis=1).reindex(w.index).mul(cov_chara_ret.loc[t_range_out_of_sample].stack().reindex(w.index),axis=0).groupby(level=0).sum()
PortfolioAnalysis(ret_1_n)
(w.loc[:,:].mean(axis=1).unstack().std()==0.0).sum()
w.loc[:,:].mean(axis=1).unstack()#.std()
# 权重的动态变化



# todo Parametric Portfolio Policy（先选取因子），样本内表现；结果对predictors的选择、gam参数的奇偶比较敏感
from scipy.optimize import minimize,Bounds,LinearConstraint,NonlinearConstraint,BFGS,SR1
#np.abs(PortfolioAnalysis(cov_chara_ret).loc['SR']).sort_values(ascending=False)
# 结果不是很好SR=0.229655
#predictors_names = ('''BM,EP,PY,CHMOM,REV,MOM6M,IVOL,ILLIQ,MAXRET,RVOL,SIZE,STD_RVOL,STD_TURN,RETVOL,TURN''').split(',');predictors_names
# 结果还行; ROA似乎既没优化，也没恶化;SG优化；
predictors_names = ('''SIZE,BM,SG,REV,IVOL,ILLIQ,RVOL,RETVOL,STD_TURN,TURN''').split(',');predictors_names
# 使用所有的变量会使得结果变得非常差
#predictors_names=signal_names.split(',')
ret_chara=cov_chara_ret.loc[:'2019-11',predictors_names].apply(lambda x:x.fillna(x.mean())*np.sign(x.mean()))
# 加入 benchmark
ret_benchmark,weights_bench=GenerateMarketPortfolioReturnsAccordingInvestingDate(ret)
ret_bench=ret_benchmark.loc[:'2019-11']*100.0
gam=3.0 # 结果对风险厌恶参数比较敏感！！！
def UtilityFunction(x):
    return x**(1.0-gam)/(1.0-gam)
def UtilityFunctionDer(x):
    return x**(-gam)
def UtilityFunctionDer2(x):
    return -gam*x**(-gam-1.0)
def Function(theta):
    f=0.0
    for t in ret_chara.index:
        f-=UtilityFunction(ret_bench.loc[t]+ret_chara.loc[t]@theta)
        #f -= UtilityFunction(ret_chara.loc[t] @ theta)
    return f/len(ret_chara)
def FunctionDer(theta):
    f=0.0
    for t in ret_chara.index:
        f-=UtilityFunctionDer(ret_bench.loc[t]+ret_chara.loc[t]@theta)*ret_chara.loc[t]
        #f -= UtilityFunctionDer(ret_chara.loc[t] @ theta) * ret_chara.loc[t]
    return f/len(ret_chara)
def FunctionHess(theta):
    f=0.0
    for t in ret_chara.index:
        tmp=pd.DataFrame(ret_chara.loc[t])
        f-=UtilityFunctionDer2(ret_bench.loc[t]+ret_chara.loc[t]@theta)*(tmp.values@tmp.values.T)
        #f -= UtilityFunctionDer2(ret_chara.loc[t] @ theta) * (tmp.values @ tmp.values.T)
    return f/len(ret_chara)
#matrix_LC=pd.DataFrame(np.ones(ret_chara.shape[1])).T.values
#linear_constraint=LinearConstraint(matrix_LC, 0.99, 2.0) # 加上市场组合后似乎不需要考虑约束问题
theta0=np.ones(ret_chara.shape[1])/ret_chara.shape[1]
# 从结果来看，纯特征组合的效果还是很不错的
res=minimize(Function,theta0,method='trust-constr',jac=FunctionDer,hess=FunctionHess)#,constraints=linear_constraint)# 似乎要设置jac矩阵
# 特征组合的表现
PortfolioAnalysis((cov_chara_ret[predictors_names].apply(lambda x:x.fillna(0.0)*np.sign(x.mean()))@(res.x)).replace(0.0,np.nan))
# 特征组合与benchmark的融合组合表现【与不加bench的纯多空组合结果相近】
PortfolioAnalysis((cov_chara_ret[predictors_names].apply(lambda x:x.fillna(0.0)*np.sign(x.mean()))@(res.x)).replace(0.0,np.nan)+ret_benchmark) # 样本内效果似乎没有Sparse PCA效果好；gam参数？线性约束错误？组合/参数数量太多？？？
w1=chara_port_standard[ret_chara.columns].mul(pd.Series(res.x,index=ret_chara.columns),axis=1).sum(axis=1).unstack()+weights_bench
(w1<0.0).sum(axis=1)/(~w1.isnull()).sum(axis=1)
# todo 非常奇怪的一点，下面这个结论出现多次，但却是针对不同的设置
'''
mean            0.712320
std             1.131557
SR              0.629505
t               9.731905
t_NW_adjusted  10.035430
'''
weights_stock=GenerateStockWeights(res.x,chara_port_standard[predictors_names],w_bench=weights_bench)
weights_stock.loc['2019-11-01'].sum()
# 对比1/N 策略:表现没有PPP好
PortfolioAnalysis(ret_chara.mean(axis=1))
'''
mean           0.505443
std            0.970306
SR             0.520910
t              8.053076
t_NW_adjusted  6.631009
'''



# todo PPP+PCA 样本外
# 在时间点t，先计算历史a个PCA组合（依t值进行筛选），再计算相应的ppp策略
def WeightsEstimatedFromPCA(ret_p,n_com=15):
    # 删除还是填充
    #ret_port=ret_p.reindex(columns=ret_p.columns[ret_p.isnull().mean()<0.5])
    ret_port=ret_p.apply(lambda x:x.fillna(x.mean())).dropna(how='all',axis=1)
    s,V=np.linalg.eigh(ret_port.cov())
    ret_port_reconstrcut = (ret_port.fillna(0.0) @ V[:, :]).replace(0.0, np.nan)
    for i in range(V.shape[1]):
        # 注意，权重符号已经调整完毕
        V[:, i] = V[:, i] * np.sign(ret_port_reconstrcut[i].mean())
    ret_port_reconstrcut = (ret_port.fillna(0.0) @ V[:, :]).replace(0.0, np.nan)
    PortfolioAnalysis(ret_port_reconstrcut)
    # 根据组合的组合的平均收益，调整组合的符号
    weights = pd.DataFrame(V, index=ret_port.columns);
    # 按t值选，还是按SR选择[实证结果显示按t值排序效果更好]
    select_port=np.abs(PortfolioAnalysis(ret_port_reconstrcut)).T.sort_values(by='t_NW_adjusted',ascending=False).index[:n_com]
    return weights[select_port]#.apply(lambda x:x/np.abs(x).sum())
def GenerateWeightsFromPPP(ret_p,ret_b,w,gam):
    def UFunc(x,gam):
        return x**(1.0-gam)/(1.0-gam)
    def UFuncDer(x,gam):
        return x**(-gam)
    def UFuncDer2(x,gam):
        return -gam*x**(-gam-1.0)
    def TFunc(theta):
        f=0.0
        for t in ret_p.index:
            f-=UFunc(ret_b.loc[t]+ret_p.loc[t]@theta,gam)
            #f -= UtilityFunction(ret_chara.loc[t] @ theta)
        return f/len(ret_p)
    def TFuncDer(theta):
        f=0.0
        for t in ret_p.index:
            f-=UFuncDer(ret_b.loc[t]+ret_p.loc[t]@theta,gam)*ret_p.loc[t]
        #f -= UtilityFunctionDer(ret_chara.loc[t] @ theta) * ret_chara.loc[t]
        return f/len(ret_p)
    def TFuncHess(theta):
        f=0.0
        for t in ret_p.index:
            tmp=pd.DataFrame(ret_p.loc[t])
            f-=UFuncDer2(ret_b.loc[t]+ret_p.loc[t]@theta,gam)*(tmp.values@tmp.values.T)
            #f -= UtilityFunctionDer2(ret_chara.loc[t] @ theta) * (tmp.values @ tmp.values.T)
        return f/len(ret_p)
    theta0=np.ones(ret_p.shape[1])/ret_p.shape[1]
    res=minimize(TFunc,theta0,method='trust-constr',jac=TFuncDer,hess=TFuncHess)#,constraints=linear_constraint)# 似乎要设置jac矩阵
    return (w.fillna(0.0)@(res.x))
ret_benchmark,weights_bench=GenerateMarketPortfolioReturnsAccordingInvestingDate(ret)
ret_bench=ret_benchmark.loc[:'2019-11']*100.0
n_com=15
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
w1=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product([t_range_out_of_sample,signal_names.split(',')]),columns=range(n_com))
w2=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=signal_names.split(','))
for t in t_range_out_of_sample[5:]:
    w1.loc[t]=WeightsEstimatedFromPCA(cov_chara_ret.loc[:t-Day()],n_com).reindex(index=cov_chara_ret.columns).values
    #ret_chara=cov_chara_ret.loc[:t-Day()].fillna(0.0)@w.loc[t]
    w2.loc[t]=GenerateWeightsFromPPP(cov_chara_ret.loc[:t-Day()].fillna(0.0)@w1.loc[t],ret_bench,w1.loc[t],3.0)
#w2.to_pickle(data_path_local+'PPP_PCA_weights')
# 样本外结果
PortfolioAnalysis((w2*cov_chara_ret.reindex(w2.index)).mean(axis=1).loc['2010':])
# 样本内结果，不好！！！
PortfolioAnalysis(cov_chara_ret.loc[:'2019-11'].fillna(0.0)@w2.loc['2019-11'].values.T)
'''
mean           0.318254
std            1.363330
SR             0.233439
t              3.608881
t_NW_adjusted  2.918760
'''

# todo 样本内：mean-variance PPP + 稀疏特征
predictors_names = ('''SIZE,BM,SG,REV,IVOL,ILLIQ,RVOL,RETVOL,STD_TURN,TURN''').split(',');predictors_names
ret_chara=cov_chara_ret.loc[:'2019-11',predictors_names].apply(lambda x:x.fillna(x.mean())*np.sign(x.mean()))
# 1/N 策略 表现不如PPP
PortfolioAnalysis(ret_chara.mean(axis=1),winso=True)
# 没有shrinkage
PortfolioAnalysis(ret_chara.fillna(0.0)@(np.linalg.inv(ret_chara.cov())@ret_chara.mean()))
# shrinkage
kappa=10000000.0 # 0.8 #
gam=np.matrix.trace(ret_chara.cov().values)/len(ret_chara.loc[:'2019-11'])/kappa**2.0
tmp_is=\
    PortfolioAnalysis((ret_chara.fillna(0.0)@(np.linalg.inv(ret_chara.cov()+np.diag(gam*np.ones(len(predictors_names))))@ret_chara.mean())).replace(0.0,np.nan))
#tmp_is.to_excel(data_path_local+'PPP_Spasity_IS.xlsx')
# todo 样本外：mean-variance PPP + 稀疏特征
# 系数特征时，似乎并不需要shrinkage
kappa=10000000.0 # 0.8 #  越大，倾向于不shrink
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
w=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=predictors_names)
for t in t_range_out_of_sample:
    gam = np.matrix.trace(ret_chara.loc[:t-Day()].cov().values) / len(ret_chara.loc[:t-Day()]) / kappa ** 2.0
    w.loc[t]=np.linalg.inv(ret_chara.loc[:t-Day()].cov()+np.diag(gam*np.ones(len(predictors_names))))@ret_chara.loc[:t-Day()].mean()
tmp_oos=\
PortfolioAnalysis(((ret_chara.reindex(t_range_out_of_sample).fillna(0.0)*w.apply(lambda x:x/x.abs().sum(),axis=1)).sum(axis=1).replace(0.0,np.nan)))
tmp_oos.to_excel(data_path_local+'PPP_Spasity_OOS.xlsx')


# todo 样本内：mean-variance 框架下的PPP + factor zoo ---> PCA+shirinkage
# 步骤：PCA 分解，theta=D^(-1)*mu
# PCA方法生成的权重矩阵有个问题：权重和不为0（平方和为0，因为单位向量）
#gam=3.0 # 1/N策略对应何种风险厌恶
kappa=10000000.0 #1.0# 0.8 # kappa越大意味着没有shrinkage；样本内没有shrinkage时，效果更好
gam=np.matrix.trace(cov_chara_ret.cov().values)/len(cov_chara_ret.loc[:'2019-11'])/kappa**2.0
n_com=57# 样本内，无论是否shrinkage时，n越大越好
D,V_T=np.linalg.eigh(cov_chara_ret.cov())
#(cov_chara_ret.fillna(0.0)@V_T[:,:15]).replace(0.0,np.nan).plot();plt.show()
ret_port_reconstrcut=(cov_chara_ret.fillna(0.0)@V_T[:,-n_com:]).replace(0.0,np.nan)
# gam 越小，效果越好；不需要事先调整符号
#tmp_is=\
PortfolioAnalysis(ret_port_reconstrcut@(np.diag(1.0/(D[-n_com:]+gam))@ret_port_reconstrcut.mean()),winso=True)
#tmp_is.to_excel(data_path_local+'PPP_PCA_IS_without_shrinkage.xlsx')
# 通过特征组合的权重还原最终组合
PortfolioAnalysis((cov_chara_ret.fillna(0.0)@V_T[:,-n_com:]@(np.diag(1.0/(D[-n_com:]+gam))@ret_port_reconstrcut.mean())).replace(0.0,np.nan))
# 直接 利用特征组合协方差和样本均值进行特征组合的权重估计，结果与n_com=57的情况类似
PortfolioAnalysis((cov_chara_ret.fillna(0.0)@(np.linalg.inv(cov_chara_ret.cov()+np.diag(gam*np.ones(57)))@cov_chara_ret.mean())).replace(0.0,np.nan))
# 特征组合权重
pd.Series(np.linalg.inv(cov_chara_ret.cov()+np.diag(gam*np.ones(57)))@cov_chara_ret.mean(),index=cov_chara_ret.columns).abs().sum()

# todo 样本外：mean-variance 框架下的PPP + factor zoo ---> PCA+shirinkage
# todo shrinkage虽然可以减少误差的影响，但是，无法达到sparsity的目的
def WeightsEstimatedFromPPP_PCA(ret_p,kappa=1.0,n_com=57):
    # 删除还是填充
    #ret_port=ret_p.reindex(columns=ret_p.columns[ret_p.isnull().mean()<0.5])
    ret_port=ret_p.apply(lambda x:x.fillna(x.mean())).dropna(how='all',axis=1)
    gam=np.matrix.trace(ret_port.cov().values)/len(ret_port)/kappa**2.0
    D,V=np.linalg.eigh(ret_port.cov())
    ret_port_reconstrcut = (ret_port.fillna(0.0) @ V[:, -n_com:]).replace(0.0, np.nan)
    return V[:, -n_com:] @ (np.diag(1.0 / (D[-n_com:] + gam)) @ ret_port_reconstrcut.mean())
n_com=57# todo 不同设置结果不同：【方案1】kappa=100000，n=15,好；【方案2】有shrinkage时，n=57似乎是最好的；
#gam=3.0 # 0.0 的样本外效果反而差一些
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
#ret_top_ports=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=range(1))
w=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=signal_names.split(','))
for t in t_range_out_of_sample:
    # shrinkage效果更好
    w.loc[t]=WeightsEstimatedFromPPP_PCA(cov_chara_ret.loc[:t-Day()],kappa=1.0,n_com=n_com)#.reindex(index=cov_chara_ret.columns).values
# 似乎在n_com较小时，使用expanding会优化结果
ret_optimal_port=(cov_chara_ret.loc[t_range_out_of_sample]*w.apply(lambda x:x/x.abs().sum(),axis=1)).sum(axis=1)# w.expanding().mean()
# 组合的自相关性非常强；剔除异常值会使得结果更好
#tmp_oos=\
PortfolioAnalysis(ret_optimal_port,winso=True)#.loc['t_NW_adjusted'].sort_values()
#tmp_oos.to_excel(data_path_local+'PPP_PCA_OOS_without_shrinkage.xlsx')
# 画图
ret_optimal_port.plot(label='PPP+PCA+OOS');
plt.plot(ret_optimal_port.index,np.zeros(len(ret_optimal_port.index)))
plt.legend();plt.show()
# 在不shrinkage下，n_com与SR的关系
def RelationBetween_N_SR_kappa(n_com=57,kappa=1.0):
    w = pd.DataFrame(np.nan, index=t_range_out_of_sample, columns=signal_names.split(','))
    for t in t_range_out_of_sample:
        # shrinkage效果更好
        w.loc[t] = WeightsEstimatedFromPPP_PCA(cov_chara_ret.loc[:t - Day()], kappa=kappa,
                                               n_com=n_com)  # .reindex(index=cov_chara_ret.columns).values
    # 似乎在n_com较小时，使用expanding会优化结果
    ret_optimal_port = (cov_chara_ret.loc[t_range_out_of_sample] * w.apply(lambda x:x/x.abs().sum(),axis=1)).sum(axis=1)  # w.expanding().mean()
    return PortfolioAnalysis(ret_optimal_port, winso=True).loc[(['SR','SR_ann'])]
SR_=pd.DataFrame(np.nan,index=np.arange(1,58),columns=['SR','SR_ann'])
for i in SR_.index:
    SR_.loc[i]=\
        RelationBetween_N_SR_kappa(n_com=i,kappa=100000.0)[0]

SR_.to_excel(data_path_local+'relation_N_SR.xlsx')
SR_=pd.read_excel(data_path_local+'relation_N_SR.xlsx')[['SR','SR_ann']]
plt.plot(pd.Series(SR_.index)+1,SR_['SR'],label='SR'); #,
plt.plot(pd.Series(SR_.index)+1,SR_['SR_ann'],label='SR_ann')
plt.legend();plt.show()


# todo 加权估计协方差+Sparse PCA
# 1、给定某个投资时间的
def WeightedCovariance(ret_port,halflife=None):
    halflife=int(len(ret_port)/2.0) # 这样设置的结果不够理想，问题再halflife还是Sparse PCA的输入不该是cov？？？
    lam=(0.5)**(1.0/halflife)
    w=pd.Series(lam**np.arange(len(ret_port)-1,-1,-1),index=ret_port.index)
    w/=w.sum()
    ret_port_adj=(ret_port-ret_port.mean()).fillna(0.0).mul(np.sqrt(w),axis=0)
    cov_matrix=pd.DataFrame(0.0,index=ret_port_adj.columns,columns=ret_port_adj.columns)
    for t in ret_port_adj.index:
        cov_matrix+=pd.DataFrame(ret_port_adj.loc[t]).values@pd.DataFrame(ret_port_adj.loc[t]).T.values
    return cov_matrix
def WeightsEstimatedFromSparsePCAWithWeightedCovariance(ret_p,n_com=30):
    ret_port=ret_p.dropna(how='all',axis=1)
    tf = SparsePCA(n_components=n_com)  # , random_state=0)
    cov_matrix=WeightedCovariance(ret_port)
    tf.fit(cov_matrix)  # 注意量级
    tf.transform(ret_port.fillna(0.0)) # .apply(lambda x:x.where(~x.isnull(),x.mean()),axis=0))#,index=date_investing[date_investing<'2019-12'])
    # 根据组合的组合的平均收益，调整组合的符号
    weights = pd.DataFrame(tf.components_, columns=cov_matrix.columns).T;
    ret_transformed_port =(ret_port.fillna(0.0) @ weights).replace(0.0, np.nan)
    for c in ret_transformed_port.columns:
        weights[c] = weights[c] * np.sign(ret_transformed_port[c].mean()) / np.abs(weights[c]).sum()
    ret_transformed_port = (ret_port.fillna(0.0) @ weights).replace(0.0, np.nan);
    # 按t值选，还是按SR选择
    select_port=np.abs(PortfolioAnalysis(ret_transformed_port)).T.sort_values(by='SR',ascending=False).index
    for p in select_port:
        weights[p]*=np.sign(ret_transformed_port[p].mean())
    return weights[select_port]
n_com=25
t_range_out_of_sample=date_investing[(date_investing>'2005-01') & (date_investing<'2019-12')]
#ret_optimal_port=pd.DataFrame(np.nan,index=t_range_out_of_sample,columns=range(1))
w=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product([t_range_out_of_sample,signal_names.split(',')]),columns=range(n_com))
for t in t_range_out_of_sample:
    w.loc[t]=\
        WeightsEstimatedFromSparsePCAWithWeightedCovariance(100.0*cov_chara_ret.loc[t-DateOffset(months=120):t-Day()],n_com).values
ret_optimal_port=w.mul(cov_chara_ret.loc[t_range_out_of_sample].stack().reindex(w.index),axis=0).groupby(level=0).sum()
PortfolioAnalysis(ret_optimal_port).T.sort_values(by='t_NW_adjusted',ascending=False)
# 1/n 策略
ret_1_n=w.loc[:,:11].mean(axis=1).reindex(w.index).mul(cov_chara_ret.loc[t_range_out_of_sample].stack().reindex(w.index),axis=0).groupby(level=0).sum()
PortfolioAnalysis(ret_1_n)
(w.loc[:,:].mean(axis=1).unstack().std()==0.0).sum()
w.loc[:,:].mean(axis=1).unstack()#.std()




# todo EWMA+同频历史数据




# todo PLS 配合winddb下的数据
# 第一步：R~x上，共A个截面回归
# 由于特征已经经过标准化处理，所以beta系数就等于Z*r/N
from scipy import stats
#chara_port_standard_winsorize=chara_port_standard.groupby(level=0).apply(lambda x:x.apply(lambda y:stats.mstats.winsorize(y,limits=[0.01,0.01]),axis=0))#.groupby(level=0).agg(['min','max'])
cov_chara_ret=100.0*chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).mean();
# 记f为特征组合收益率，计算inv(f'*f)，注意shift(1)
ff=1.0/cov_chara_ret.apply(lambda x:(x**2.0).mean(),axis=1).replace(0.0,np.nan).shift(1)
# 第二步：计算f'*Z，注意f要shift(1) # 假定ff固定时，结果很好！！！
f=cov_chara_ret.shift(1)
f_rolling_12=f.rolling(window=12).mean()
f_expanding=f.expanding().mean()
fZ_without_rolling=chara_port_standard.apply(lambda x:f.loc[x.name[0]].mul(x.reindex(f.columns)).mean(),axis=1).mul(ff)
fZ_with_12M_rolling=chara_port_standard.apply(lambda x:f_rolling_12.loc[x.name[0]].mul(x.reindex(f.columns)).mean(),axis=1).mul(ff)
fZ_with_expanding=chara_port_standard.apply(lambda x:f_expanding.loc[x.name[0]].mul(x.reindex(f.columns)).mean(),axis=1).mul(ff)

qcut_options = {'q': np.arange(0, 1.00001, 0.2), 'q_con': np.arange(0, 1.00001, 0.2)}
#qcut_options = {'q': (0.0, 0.3, 0.7, 1.0), 'q_con': (0.0, 0.5, 1.0)};
qcut_options['labels'] = np.arange(1, len(qcut_options['q']));
qcut_options['labels_con'] = np.arange(1, len(qcut_options['q_con']))

p_without_rolling=Returns_UnivariatePortfolio(fZ_without_rolling.unstack().reindex(index=date_investing.iloc[1:-1],columns=ret.columns)[filtered_stock],ret.loc['2000-02':'2019-11']*100.0,qcut_options)#,weights=SIZE.loc['2000-02':'2019-11'])
p_with_12M_rolling=Returns_UnivariatePortfolio(fZ_with_12M_rolling.unstack().reindex(index=date_investing.iloc[1:-1],columns=ret.columns)[filtered_stock],ret.loc['2000-02':'2019-11']*100.0,qcut_options)#,weights=SIZE.loc['2000-02':'2019-11'])
p_with_expanding=Returns_UnivariatePortfolio(fZ_with_expanding.unstack().reindex(index=date_investing.iloc[1:-1],columns=ret.columns)[filtered_stock],ret.loc['2000-02':'2019-11']*100.0,qcut_options)#,weights=SIZE.loc['2000-02':'2019-11'])
# 很有意思的一点现象：分位组合相关性非常高，且波动率相近；得分最高的分位组合显著，但多空组合尤其显著
tmp1=PortfolioAnalysis(pd.concat([p_without_rolling,p_without_rolling.iloc[:,-1]-p_without_rolling.iloc[:,0]],axis=1))
tmp2=PortfolioAnalysis(pd.concat([p_with_12M_rolling,p_with_12M_rolling.iloc[:,-1]-p_with_12M_rolling.iloc[:,0]],axis=1))
tmp3=PortfolioAnalysis(pd.concat([p_with_expanding,p_with_expanding.iloc[:,-1]-p_with_expanding.iloc[:,0]],axis=1))
#tmp1.to_csv(data_path_local+'AFER_without_rolling.csv')
#tmp2.to_csv(data_path_local+'AFER_with_12M_rolling.csv')
#tmp3.to_csv(data_path_local+'AFER_with_expanding.csv')
pd.concat([p_with_12M_rolling,p_with_12M_rolling.iloc[:,-1]-p_with_12M_rolling.iloc[:,0]],axis=1).cov().to_excel(data_path_local+'cov.xlsx')
# 三因子alpha：alpha主要来源于空头
port_market=GenerateMarketPortfolioReturnsAccordingInvestingDate(ret,w=SIZE)
port_market=port_market.sub(np.exp(interest_rate_free*.01)-1.0,axis=0)
smb_hml=pd.read_pickle(data_path_local+'SMB_HML_'+freq.upper())
X=pd.concat([port_market,smb_hml],axis=1).loc['2001-01':'2019-11']*100.0
XXX=np.linalg.inv(X.agg(lambda x:x-x.mean()).T@X.agg(lambda x:x-x.mean()))@X.agg(lambda x:x-x.mean()).T#@port_long_short.loc[:'2019-11'].fillna(port_long_short.mean())
def GenerateAlphaAndErrorByOLS(Y,X,XXX):
    beta=XXX@Y.agg(lambda x:x-x.mean()).fillna(0.0)
    ee=Y-X.values@beta
    return ee
e_without_rolling=\
    pd.concat([p_without_rolling,p_without_rolling.iloc[:,-1]-p_without_rolling.iloc[:,0]],axis=1).loc['2001':'2019-11'].apply(lambda y:GenerateAlphaAndErrorByOLS(y,X,XXX))
e_with_12M_rolling=\
    pd.concat([p_with_12M_rolling,p_with_12M_rolling.iloc[:,-1]-p_with_12M_rolling.iloc[:,0]],axis=1).loc['2001':'2019-11'].apply(lambda y:GenerateAlphaAndErrorByOLS(y,X,XXX))
e_with_expanding=\
    pd.concat([p_with_expanding,p_with_expanding.iloc[:,-1]-p_with_expanding.iloc[:,0]],axis=1).loc['2001':'2019-11'].apply(lambda y:GenerateAlphaAndErrorByOLS(y,X,XXX))
e_without_rolling.mean()/e_without_rolling.apply(lambda y:NWest(y,X))
e_with_12M_rolling.mean()/e_with_12M_rolling.apply(lambda y:NWest(y,X))
e_with_expanding.mean()/e_with_expanding.apply(lambda y:NWest(y,X))

def GenerateResults(tmp,e):
    res=pd.DataFrame(np.nan,index=['mean','t','alpha','t_alpha','SR'],columns=tmp.columns)
    res.loc[(['mean','t','SR'])]=tmp.loc[(['mean','t_NW_adjusted','SR'])].values
    res.loc[(['alpha','t_alpha'])]=e.mean(),e.mean()/e.apply(lambda y:NWest(y,X))
    return res
GenerateResults(tmp1,e_without_rolling).to_excel(data_path_local+'AFER_without_rolling_VW.xlsx')
GenerateResults(tmp2,e_with_12M_rolling).to_excel(data_path_local+'AFER_with_12M_rolling_VW.xlsx')
GenerateResults(tmp3,e_with_expanding).to_excel(data_path_local+'AFER_with_expanding_VW.xlsx')

PortfolioAnalysis(ret.mean(axis=1))



# todo 中国版三因子，剔除30%的股票


# todo 行业中性化，降低组合风险