




import pandas as pd
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg



closes  = pd.read_csv('./SnP_holdings_random_pick.csv', index_col=0 )
closes_return = closes.pct_change().dropna()
closes_return = closes_return.reset_index()
closes_return['Date'] = pd.to_datetime(closes_return['Date'])
closes_return = closes_return.set_index('Date')

#
##### the strategy is based on a monthly allocation and 3M look-back covar
lookBackDays=70
alloc_period='M'
quantile_ = 0.1

num_assets = closes.shape[1]-1 ## -1 because one col is the spy (index)
########### var of spy
spy_vol_3M1M = closes_return['SPY'].rolling(lookBackDays).std().resample(alloc_period).mean().dropna()
########### cov matrixes
cov_mtx_time_series = closes_return.rolling(lookBackDays).cov().dropna()
idx_start_date = np.where(closes_return.index == cov_mtx_time_series.index[0][0])[0][0]
total_dates_in_series=(closes_return.shape[0] -idx_start_date)

spy_index= np.where(cov_mtx_time_series[0:1].columns=='SPY')[0][0]
betas = np.zeros([total_dates_in_series, num_assets])
for i in range(total_dates_in_series):
    cov = cov_mtx_time_series[(i*(num_assets+1)):((i+1)*(num_assets+1))]
    betas[i, :] = list(cov.values[spy_index, :spy_index]) + list(cov.values[spy_index, (spy_index+1):])

cols=list(cov.columns[:spy_index])+list(cov.columns[(spy_index+1):])
betas = pd.DataFrame(betas, 
                     index = closes_return.index[idx_start_date:],
                     columns=cols)
betas = betas.resample(alloc_period).mean()
for r in range(betas.shape[0]):
    betas.iloc[r]= betas.iloc[r].apply(lambda x: x/spy_vol_3M1M.iloc[r])


########## assignment of weghts
########## in each allocation period the assets with a beta higher than cutoff 
########## of that period receives a w=1 and otherwise w=0

weights = np.zeros(betas.shape, dtype=int)
for r in range(betas.shape[0]):
    cutoff  = betas.iloc[r].quantile(quantile_)
    weights[r, :]  = betas.iloc[r].apply(lambda x: (0,1)[x>cutoff])

weights_df = pd.DataFrame(weights, index= betas.index, columns=cols)


########## application to the returns 
### lets exclude the SPY from returns 
portfolio_return = np.zeros([betas.shape[0]-1, 1])
closes_return_assets = closes_return.drop(['SPY'], axis=1)
for alloc_idx in range(len(betas.index)-1):
    start_date = betas.index[alloc_idx]
    end_date   = betas.index[alloc_idx+1]
    period_returns =closes_return_assets.loc[start_date:end_date]
    total_per_date_in_period_return = period_returns.mul(weights[alloc_idx, :], ).sum(axis=1)
    # full return in alloc_period
    portfolio_return[alloc_idx,0]=(total_per_date_in_period_return + 1.0).prod() - 1 
    
col_name = 'portfolio %s '%(alloc_period) + 'Return %'
portfolio_return=pd.DataFrame(portfolio_return, index=betas.index[1:], columns=[col_name ])

portfolio_return.plot(style='--o')
    
print '========================================================='
print 'alloc_period : ', alloc_period
print 'hist data used from %d days prior : '%lookBackDays
print 'allocation to quantiles > %f : '%quantile_
print 'Total return over the horizon of the strategy %:', portfolio_return.sum()[0]
print '========================================================= \n\n'

#################################  The index return over same period:

market_index = closes_return['SPY']
return_per_period_idx=np.zeros([betas.shape[0]-1, 1])
for alloc_idx in range(len(betas.index)-1):
    start_date = betas.index[alloc_idx]
    end_date   = betas.index[alloc_idx+1]
    period_returns_idx = market_index.loc[start_date:end_date]
    return_per_period_idx[alloc_idx, 0] = (period_returns_idx+1).prod()-1


col_name = 'SPY %s '%(alloc_period) + 'Return %'
return_per_period_idx=  pd.DataFrame(return_per_period_idx, index=betas.index[1:], columns=[col_name ])

return_per_period_idx.plot(style='--o')

    
print '========================================================='   
print 'alloc_period : ', alloc_period
#print 'hist data used from %d days prior : '%lookBackDays
print 'Total return of SPY %:', return_per_period_idx.sum()[0]
print '========================================================='
    


