import pandas as pd
from FactorZooUsingSQL import *
import matplotlib.pyplot as plt
import numpy as np
from spca.spca import compute_spca_brute_force

data_path_local =  '/Users/harbes/PycharmProjects/data/winddb/' #'E:/data/winddb/' #
signal_names = '''AM,BM,CFP,DER,DLME,EP,LG,PY,SG,SMI,SP,TG,ACC,PACC,CAPXG,dBe,dPIA,IA,IVC,IVG,NOA,ATO,CTO,EY,GM,GP,NPOP,RNA,ROA,ROE,TBI,Z_score,CHMOM,INDMOM,REV,MOM12M,MOM6M,B_DIM,B_DN,BETA,BETASQ,B_FP,IVOL,ILLIQ,MAXRET,RVOL,SIZE,STD_RVOL,STD_TURN,RETVOL,TURN,CFD,CR,CRG,QR,QRG,SI'''

def GenerateDailyMarketPortfolioReturns(ret,size=None):
    if size is None:
        return ret.mean(axis=1)
    else:
        return (ret*size).sum(axis=1)/((~ret.isnull())*size).sum(axis=1)


#trading_calendar = GetTradingCalendarFromSQL()
trading_calendar=pd.read_pickle(data_path_local+'trading_calendar')
#ashare_stock = GetAshareStockCodeFromSQL()
ashare_stock=pd.read_pickle(data_path_local+'ashare_stock_all')
# filtered_stock=GetFilter(del_finance=False).reindex(date_investing)
freq_list = ['M', '2W', 'W']
freq=freq_list[0]


#date_investing = GetInvestingDate(freq=freq)
date_investing=pd.read_pickle(data_path_local+'date_investing_'+freq)
filtered_stock=GetFilter(del_finance=True).reindex(date_investing)
ret = GenerateStockFutureReturns(date_investing,ashare_stock)  # [filtered_stock] # 每个investing date对应的收益率r_{t+1}
SIZE = Size(date_investing, ashare_stock)  # [filtered_stock] # EW:-3.415907 ; VW: -2.863930
predictors_all=pd.read_pickle(data_path_local + 'predictors_all_' + freq).replace(np.inf,np.nan)
ret_stock_d=pd.read_pickle(data_path_local+'adj_prc').pct_change().replace([np.inf,0.0],np.nan)
# 计算 特征组合的收益
# 检查数据维度，
ret.size # 926640
ret.stack().size # 465162
len(predictors_all) # 483701

# 特征标准化
chara_port_standard=predictors_all.groupby(level=0).apply(lambda x:(x-x.mean())/x.std())
# 检查是否标准化
#chara_port_standard.groupby(level=0).agg(['mean','std'])
# 特征组合收益率
ret_chara_port=chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).sum();ret_chara_port
# 特征与截面ret之间的协方差【由于标准化，所以cov=rho=beta(ret~chara)；特征与截面ret之间的相关系数=cov(chara,ret)/disp(chara)/disp(ret)】
cov_chara_ret=chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).mean();cov_chara_ret
# beta(chara~ret)
#beta_chara_ret=chara_port_standard.mul(ret.stack().reindex(chara_port_standard.index),axis=0).groupby(level=0).mean()/ret.std(axis=1)

(ret_chara_port.std(axis=1)/ret_chara_port.std(axis=1).mean()).plot(label='ret_chara_port_disp');
(cov_chara_ret.std(axis=1)/cov_chara_ret.std(axis=1).mean()).plot(label='cov_disp');
#(ret.std(axis=1)/ret.std(axis=1).mean()).plot(label='ret_disp')
plt.legend()
plt.show()

cov_chara_ret.std(axis=1).idxmax()
cov_chara_ret.loc[cov_chara_ret.std(axis=1).idxmax()]
# 不同时期最有效的预测变量
cov_chara_ret.agg(['idxmin','idxmax'],axis=1)

# PCA
#from sklearn.decomposition import PCA
#num_pc=57
#pca = PCA(n_components=num_pc)
#pca.fit(cov_chara_ret)
#x_pca = pd.DataFrame(pca.fit(ret_chara_port).transform(ret_chara_port),index=ret.index,columns=range(1,num_pc+1))
#x_pca.plot();plt.show()
#PortfolioAnalysis(x_pca.loc[:,10:]).loc['t_NW_adjusted']#.idxmax()

# PCA using numpy
# todo 如何选择reconstructed portfolio？ Sharpe ratio？
# PCA方法生成的权重矩阵有个问题：权重和不为0（平方和为0，因为单位向量）
tmp=(cov_chara_ret.corr()+cov_chara_ret.corr().T)/2.0
s,V_T=np.linalg.eigh(tmp)
#(cov_chara_ret.fillna(0.0)@V_T[:,:15]).replace(0.0,np.nan).plot();plt.show()
ret_port_reconstrcut=(cov_chara_ret.fillna(0.0)@V_T[:,:]).replace(0.0,np.nan)
# 调整符号
for i in range(V_T.shape[1]):
    V_T[:,i]=V_T[:,i]*np.sign(ret_port_reconstrcut[i].mean())
ret_port_reconstrcut = (cov_chara_ret.fillna(0.0) @ V_T[:, :]).replace(0.0, np.nan)

select1=\
    ret_port_reconstrcut.columns[PortfolioAnalysis(ret_port_reconstrcut).loc['t_NW_adjusted']>3.0]
# 历史表现最优的几个组合
select2=np.abs(PortfolioAnalysis(ret_port_reconstrcut).loc['t_NW_adjusted']).sort_values(ascending=False).index[:20]
PortfolioAnalysis(ret_port_reconstrcut[select2])
# 历史最优组合；不同特征上赋予的权重，权重成梯队状，且一定范围内相近---》1/n策略
pd.DataFrame(np.abs(V_T[:,19]),index=signal_names.split(','),columns=['w']).sort_values(by=['w'],ascending=False)
PortfolioAnalysis(ret_port_reconstrcut[19])
ret_port_reconstrcut[19].plot();plt.show()
ret_port_reconstrcut[19].rolling(window=24).std().plot();plt.show()
# 最优的几个组合在不同特征组合上赋予的权重;由于最优的组合往往不相关，所以重组组合对各特征组合的权重也是不相关的
weights_select=np.abs(pd.DataFrame(V_T[:,select1],index=signal_names.split(','),columns=select1)).sort_values(by=select2[0],ascending=False)
weights_description=pd.DataFrame(V_T[:,select2[:16]],index=signal_names.split(','),columns=select2[:16]).sort_values(by=select2[0],ascending=False).agg(['min','max','mean','std'],axis=1)#.sort_values(ascending=False)

#ret_port_reconstrcut[select1].corr() # 为什么相关系数比较大---》通过实验，似乎是样本误差造成的？

# t>3 的组合采取 1/N 策略，结果表现很好
PortfolioAnalysis((cov_chara_ret.T.reindex(weights_description['mean'].index).mul(weights_description['mean'],axis=0)).sum())


# todo 利用EWMA+更高频历史数据估计协方差矩阵，进行样本外分析
# todo 如何判定PCA方法生成组合的稳定性
# 1、给定某个投资时间的
t_range=trading_calendar[trading_calendar<=t].iloc[-300:-1];t_range
chara_tmp=chara_port_standard.loc[t_range].unstack().reindex(index=t_range.iloc[-270:],method='ffill').stack();tmp
cov_tmp=chara_tmp.mul(ret_stock_d.loc[t_range.iloc[-270:]].stack(),axis=0).groupby(level=0).mean()
# EWMA获得协方差
# todo 注意：有指标缺失问题
cov_EWMA=cov_tmp.dropna(axis=1,how='any').ewm(halflife=90,min_periods=270).corr().loc[t_range.iloc[-1]]
_,V_tmp=np.linalg.eigh(cov_EWMA) # 完全共线？？？
# todo 应该选择什么组合？？？
w_top5=np.abs(PortfolioAnalysis(cov_tmp[cov_EWMA.columns]@V_tmp).loc['t_NW_adjusted']).sort_values(ascending=False).index[:5];w_top5
PortfolioAnalysis(cov_tmp[cov_EWMA.columns]@V_tmp[:,w_top5]).loc['t_NW_adjusted']
pd.DataFrame(V_tmp[:,w_top5],index=cov_EWMA.columns).mean(axis=1)


# todo EWMA+同频历史数据



# Sparse PCA
from sklearn.decomposition import SparsePCA
transformer = SparsePCA(n_components=30)#, random_state=0)
# todo 输入原始矩阵（＋标准化）还是输入协方差矩阵？？？
# 注意输入的变量量级不要太小,也不要太大，100比较适合（未/std）
transformer.fit(cov_chara_ret.dropna(how='all',axis=0).agg(lambda x:x-x.mean()).fillna(0.0)*10000.0)#.cov()*1e4)
transformer.transform(cov_chara_ret.dropna(how='all',axis=0).fillna(0.0))#.apply(lambda x:x.where(~x.isnull(),x.mean()),axis=0))#,index=date_investing[date_investing<'2019-12'])
# 根据组合的组合的平均收益，调整组合的符号
weights=pd.DataFrame(transformer.components_,columns=signal_names.split(',')).T;weights
ret_transformed_port=(cov_chara_ret.fillna(0.0)@transformer.components_.T).replace(0.0,np.nan);ret_transformed_port
for c in weights.columns:
    # 调整权重的符号，并且scale权重
    weights[c]=weights[c]*np.sign(ret_transformed_port[c].mean())/np.abs(weights[c]).sum()
ret_transformed_port=(cov_chara_ret.fillna(0.0)@weights).replace(0.0,np.nan);ret_transformed_port
res=PortfolioAnalysis(ret_transformed_port.dropna(how='all',axis=1)).T.sort_values(by='t_NW_adjusted',ascending=False);res

weights[res.index[:9]].replace(0.0,np.nan) # 最优组合总是配置在与交易成本(换手率、交易量、特质性波动率、最大收益率)相关的变量中
# 1/n策略
PortfolioAnalysis(cov_chara_ret.fillna(0.0)@weights[res.index[:2]].mean(axis=1))
ret_transformed_port.std()