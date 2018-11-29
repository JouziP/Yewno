'''
Author: Pej
Problem 2

'''


import pandas as pd
import datetime 
import numpy as np
#### for data
import fix_yahoo_finance as yf  

def get_data_dt(tickers, start_date, end_date):
    dt = yf.download(tickers, start_date , end_date)
    return dt 

SnP_holdings_tickers = pd.read_csv('./SnP_constituents.csv')
tickers=list(SnP_holdings_tickers['Symbol'].dropna().astype(str))
start_date = '2013-01-01'
end_date = '2018-01-01'

#### some random stocks 
seed = 5256
num_assets=20
random_pick = np.random.randint(0, len(tickers), num_assets)
tickers = ['SPY']+list(np.array(tickers)[random_pick])
SnP_holdings = get_data_dt(tickers, start_date, end_date)
close_prieces = SnP_holdings['Close']
close_prieces.to_csv('SnP_holdings_random_pick_%d.csv'%num_assets)
