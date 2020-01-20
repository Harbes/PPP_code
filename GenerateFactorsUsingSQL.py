import pandas as pd
from FactorZooUsingSQL import *


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
    options_mysql = {'server': 'localhost',
                     'user': 'root',
                     'password': '1234567890',
                     'database': 'winddb'}
    connect_mysql = ConnectMySQL(**options_mysql)
    # 全局变量
    data_path_local = 'E:/data/winddb/'
    qcut_options = {'q': np.arange(0, 1.00001, 0.2), 'q_con': np.arange(0, 1.00001, 0.2)}
    # qcut_options = {'q': (0.0, 0.3, 0.7, 1.0), 'q_con': (0.0, 0.5, 1.0)};
    qcut_options['labels'] = np.arange(1, len(qcut_options['q']));
    qcut_options['labels_con'] = np.arange(1, len(qcut_options['q_con']))

    trading_calendar = GetTradingCalendarFromSQL()
    ashare_stock = GetAshareStockCodeFromSQL()
    # filtered_stock=GetFilter(del_finance=False).reindex(date_investing)
    freq_list = ['M', '2W', 'W']

    # 常见的用于检验的变量
    ret_market_portfolio, ret_stock = GenerateDailyMarketPortfolioReturns(trading_calendar, ashare_stock,
                                                                          stock_ret_output=True)
    for f in range(3):
        freq = freq_list[f]
        predictors_all = pd.DataFrame()
        date_investing = GetInvestingDate(freq=freq)
        ret = GenerateStockFutureReturns(date_investing,
                                         ashare_stock)  # [filtered_stock] # 每个investing date对应的收益率r_{t+1}
        # 因子
        SIZE = Size(date_investing, ashare_stock)  # [filtered_stock] # EW:-3.415907 ; VW: -2.863930
        BM = BookToMarket(date_investing, ashare_stock)  # [filtered_stock]  # 波动小，收益也小;等权重效果更好
        IVOL = IdiosyncraticVolatility(date_investing, ashare_stock, ret_stock,
                                       ret_market_portfolio)  # [filtered_stock]  # 等权重下尤其显著-3.988093;市值加权下，弱一些-2.549268
        AM = AssetToMarket(date_investing, ashare_stock)  # [filtered_stock] # EW: t=1.806439,且分位组合存在显著; VW 1.080900
        CFP = OperatingCashFlowToPrice(date_investing,
                                       ashare_stock)  # [filtered_stock] # EW:2.624624 ;VW: 1.212390，分位组合边际显著
        DER = DebtToEquity(date_investing, ashare_stock)  # [filtered_stock] # EW:0.89，分位组合存在显著； VW：0.401308
        DLME = LongTermDebtToMarketEquity(date_investing,
                                          ashare_stock)  # [filtered_stock]# EW:-0.077937, 分位组合边际显著，且组合波动相近 ;VW: 0.14
        EP = EarningsToPrice(date_investing,
                             ashare_stock)  # [filtered_stock] # EW:(wind)2.622694;(cal)0.676660; VW: (wind)2.642525;(cal) 0.941958
        LG = LiabilityGrowth(date_investing, ashare_stock)  # [filtered_stock] # EW:-2.059612 ; VW: -0.880275
        PY = PayoutYield(date_investing, ashare_stock)  # [filtered_stock]  # EW:-2.525766; VW: -1.690539,分位组合边际显著
        SG = SustainableGrowth(date_investing,
                               ashare_stock)  # [filtered_stock] # EW:-0.740216,分位组合显著，倒U关系; VW: 0.424154
        SMI = SalesGrowthMinusInventoryGrowth(date_investing,
                                              ashare_stock)  # [filtered_stock]#EW:1.283801,分位显著 ; VW:0.773706
        SP = SalesToPrice(date_investing, ashare_stock)  # [filtered_stock] # EW:2.453595;VW:1.385327
        TG = TaxGrowth(date_investing, ashare_stock)  # [filtered_stock] # EW:0.912001,分位显著 ;VW:-0.536013 # todo 对结果保持怀疑
        ACC = Accruals(date_investing, ashare_stock)  # [filtered_stock]# EW: -0.301914,分位显著；VW: 0.611468
        PACC = PercentAccruals(date_investing, ashare_stock)  # [filtered_stock] #EW:-0.725025 ; VW:-0.347488
        CAPXG = CapitalExpenditureGrowth(date_investing,
                                         ashare_stock)  # [filtered_stock] #EW:-0.784630,分位显著 ; VW: -0.131139
        dBe = ChangeInShareholdersEquity(date_investing, ashare_stock)  # [filtered_stock] #EW:-0.520018 ;VW:-0.674493
        dPIA = ChangesInPPEandInventoryToAssets(date_investing,
                                                ashare_stock)  # [filtered_stock]# EW:0.942117 ;VW:0.369792
        IA = InvestmentToAssets(date_investing, ashare_stock)  # [filtered_stock] #EW:-1.282570,分位显著; VW:-0.703474
        IVC = InventoryChange(date_investing, ashare_stock)  # [filtered_stock]#EW:-0.827236 ;VW:-0.376729
        IVG = InventoryGrowth(date_investing,
                              ashare_stock)  # [filtered_stock](date_investing).reindex(columns=ashare_stock)[filtered_stock]#EW:-1.720611,分位显著 ;VW:-1.361553
        NOA = NetOperatingAssets(date_investing, ashare_stock)  # [filtered_stock]#EW:-1.899670;VW:-0.056406
        ATO = AssetTurnover(date_investing, ashare_stock)  # [filtered_stock]# EW:1.687131 ; VW:-0.137432
        CTO = CapitalTurnover(date_investing,
                              ashare_stock)  # [filtered_stock]# EW:1.615407(option-like???);VW:-1.128478
        # EBIT=ReturnOnCapital(inc_bef_tax,fin_exp,net_working,net_fixed,date_list,pub_date=None)
        EY = EarningsYield(date_investing, ashare_stock)  # [filtered_stock]#EW:1.697358(option-like?) ; VW:0.944170
        GM = GrossMargins(date_investing, ashare_stock)  # [filtered_stock] # EW:0.627702，分位显著; VW:1.001385
        GP = GrossProfitability(date_investing, ashare_stock)  # [filtered_stock] # EW:1.822651，分位显著; VW:1.065905
        NPOP = NetPayoutOverProfit(date_investing, ashare_stock)  # [filtered_stock]# EW:1.230146，分位显著; VW:0.777351
        RNA = ReturnOnOperatingAsset(date_investing, ashare_stock)  # [filtered_stock] # EW:0.403252，分位显著; VW:-0.239342
        ROA = ReturnOnAssets(date_investing, ashare_stock,
                             method='cal')  # [filtered_stock]#[read]EW:0.683188，分位显著VW:0.278190;[cal]EW:1.383680，分位显著；VW:0.526648
        ROE = ReturnOnEquity(date_investing, ashare_stock,
                             method='cal')  # [filtered_stock]#[read]EW:0.335645，分位显著VW:0.103810;[cal]EW:0.974223，分位显著；VW:0.236316
        TBI = TexableIncomeToBookIncome(date_investing,
                                        ashare_stock)  # [filtered_stock]# EW:1.674205，分位显著; VW: 0.331592
        Z_score = ZScore(date_investing, ashare_stock)  # [filtered_stock]# EW:1.083708，分位显著; VW: 0.702572
        CHMOM = ChangeIn6MonthMomentum(date_investing,
                                       ashare_stock)  # [filtered_stock]#EW:-1.607314，分位显著; VW: -2.456009 # todo 有意思
        MOM12M = Momentum(date_investing, ashare_stock)  # [filtered_stock] #EW:-0.721969，分位显著; VW: -0.732785
        REV = Reversal(date_investing, ashare_stock)  # [filtered_stock] #EW:-4.662851，分位显著; VW: -1.320881
        MOM6M = Momentum_7_12(date_investing, ashare_stock)  # [filtered_stock]  #EW:0.490052; VW:1.601683
        B_DIM = BetaDimson(date_investing, ashare_stock,ret_stock,
                           ret_market_portfolio)  # [filtered_stock] #EW:1.721818,分位显著; VW:0.298073
        B_DN = BetaDownside(date_investing,ashare_stock, ret_stock,
                            ret_market_portfolio)  # [filtered_stock] #EW:-0.396412,分位显著; VW:-0.504391
        BETA = BetaMarket(date_investing, ashare_stock, ret_stock,
                          ret_market_portfolio)  # [filtered_stock] #EW:-0.119851,分位显著; VW:-1.514148
        BETASQ = BETA ** 2.0  # EW:-1.242867,分位显著; VW:-2.209152 # todo 有意思.beta不显著，但是betaSQ显著？？？
        B_FP = BetaFP(date_investing, ashare_stock, ret_stock,
                      ret_market_portfolio)  # [filtered_stock] #EW:0.204255,分位显著; VW:-0.944521
        ILLIQ = Illiquidity(date_investing, ashare_stock)  # [filtered_stock]  #EW:3.318137; VW:4.041644
        MAXRET = MaxDailyReturn(date_investing, ashare_stock)  # [filtered_stock]  #EW:-2.396220; VW:-1.823673
        STD_RVOL = VolatilityOfTradingAmount(date_investing,
                                             ashare_stock)  # [filtered_stock]  #EW:-4.533814; VW:-1.477475
        STD_TURN = VolatilityOfTurnover(date_investing, ashare_stock)  # [filtered_stock]  #EW:-7.074734; VW:-3.603014
        RETVOL = VolatilityOfReturns(date_investing, ashare_stock)  # [filtered_stock]  #EW:-4.081799; VW:-2.762689
        TURN = Turnover(date_investing, ashare_stock)  # [filtered_stock]  #EW:-5.498887; VW:-2.812884
        CFD = CashflowToDebt(date_investing, ashare_stock)  # [filtered_stock]  #EW:1.695054,分位显著; VW:0.587085
        CR = CurrentRatio(date_investing, ashare_stock)  # [filtered_stock]  #EW:1.897474,分位显著; VW: 0.322304
        CRG = CurrentRatioGrowth(date_investing, ashare_stock)  # [filtered_stock]  #EW:-0.524383,分位显著; VW: -0.217156
        QR = QuickRatio(date_investing, ashare_stock)  # [filtered_stock]  #EW:1.636194,分位显著; VW: 0.423021
        QRG = QuickRatioGrowth(date_investing, ashare_stock)  # [filtered_stock]  #EW:0.173046,分位显著; VW: -0.107680
        SI = SalesToInventory(date_investing, ashare_stock)  # [filtered_stock]  #EW:0.803504,分位显著; VW: 0.494376
        INDMOM = IndustryMomentum(date_investing,
                                  ashare_stock)  # [filtered_stock]  #[过去六个月至过去一个月]EW:0.794906,分位显著; VW: -0.284613#[过去1个月]EW:1.510295,分位显著VW：0.565407
        RVOL = TradingAmount(date_investing, ashare_stock)  # [filtered_stock] # EW:-3.827704 ;VW:-1.658977

        # tmp=Returns_UnivariatePortfolio(IVOL.loc['2005':'2019'],ret.loc['2005':'2019'],qcut_options,weights=None)
        # tmp['long/short']=tmp.iloc[:,-1]-tmp.iloc[:,0]
        # PortfolioAnalysis(tmp)
        signal_list = [AM, BM, CFP, DER, DLME, EP, LG, PY, SG, SMI, SP, TG, ACC, PACC, CAPXG, dBe, dPIA, IA, IVC, IVG,
                       NOA,
                       ATO, CTO,
                       EY, GM, GP, NPOP, RNA, ROA, ROE, TBI, Z_score,
                       CHMOM, INDMOM, REV, MOM12M, MOM6M,
                       B_DIM, B_DN, BETA, BETASQ, B_FP,
                       IVOL, ILLIQ, MAXRET, RVOL, SIZE, STD_RVOL, STD_TURN, RETVOL, TURN,
                       CFD, CR, CRG, QR, QRG, SI]
        signal_names = '''AM,BM,CFP,DER,DLME,EP,LG,PY,SG,SMI,SP,TG,ACC,PACC,CAPXG,dBe,dPIA,IA,IVC,IVG,NOA,ATO,CTO,EY,GM,GP,NPOP,RNA,ROA,ROE,TBI,Z_score,CHMOM,INDMOM,REV,MOM12M,MOM6M,B_DIM,B_DN,BETA,BETASQ,B_FP,IVOL,ILLIQ,MAXRET,RVOL,SIZE,STD_RVOL,STD_TURN,RETVOL,TURN,CFD,CR,CRG,QR,QRG,SI'''
        for name, var in zip(signal_names.split(','), signal_list):
            predictors_all[name] = var.stack()
        predictors_all.to_pickle(data_path_local + 'predictors_all_' + freq)

