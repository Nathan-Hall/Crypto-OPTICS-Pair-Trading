import requests
import concurrent.futures
import pandas as pd
from binance.client import Client
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import json
import time
import math
import os.path

client = Client()

def get_symbols():
    info = client.get_exchange_info()
    symbols = [x['symbol'] for x in info ['symbols']]
    exclude = ['UP', 'DOWN', 'BEAR', 'BULL', 'TUSDUSDT', 'USTUSDT', 'USDCUSDT',
               'BUSDUSDT', 'EURUSDT', 'GBPUSDT', 'AUDUSDT', 'PAXGUSDT', 
               'SUSDUSDT', 'USDPUSDT']
    coins = [symbol for symbol in symbols if (symbol.endswith('USDT') 
                                 and all(s not in symbol for s in exclude))]
    return coins


def get_start_time(days_ago):
    now = dt.now()
    minus = timedelta(days=days_ago)
    start = round(((now-minus).timestamp())*1000)
    return start


def split_callpoints(data_length, start):
    data_mins = data_length * 24 * 60 // 5
    amount_of_calls = data_mins//1000 if data_mins % 1000 == 0 else (data_mins + 1000 - data_mins % 1000)//1000
    
    split = []
    b = lambda x: x*5*60*1000 #to milliseconds
    first = start
    for i in range(amount_of_calls):
        if i == amount_of_calls-1:
            limit = 1000-((amount_of_calls*1000)-data_mins)
            ms = b(limit)
        else:
            ms=b(1000)
        split.append([first, first+ms])
        first += b(1000)
        
    return split, amount_of_calls
        

def get_data(symbol, interval, split, call_no):
    #max 1000 per request so
    data=[]
    for i in range(call_no):
        first = split[i][0]
        second = split[i][1]
        url='https://api.binance.com/api/v3/klines?symbol=%s&interval=%s&startTime=%s&endTime=%s&limit=%d' % (symbol, interval, first, second, 1000)
        data_temp = requests.get(url).json()
        data.extend(data_temp)
        
    if len(data) > 1:
        df = pd.DataFrame(data)
        df = df.iloc[:, [0,2]]
        df.columns = ['Time', 'Close']
        df = df.set_index('Time')
        df.index = pd.to_datetime(df.index, unit='ms')
        df = df.astype(float)
        return df


def create_df():
    dfs = []
    relevant = get_symbols()
    data_length = 730
    start_time = get_start_time(data_length)
    interval='5m'
    t0 = time.time()
    splits, call_no = split_callpoints(data_length, start_time) #split days into 1000 call increments
    
    for index, coin in enumerate(relevant):
        dfs.append(get_data(coin, interval, splits, call_no))
        print(index, coin)
        time.sleep(13)

        
    named = pd.concat(dict(zip(relevant, dfs)), axis=1)
    t1 = time.time()
    print(t1-t0)
    
    return named


def new_data():
    print('Getting new data')
    data = create_df()
    data = data.loc[:,data.columns.get_level_values(1).isin(['Close'])]
    data.columns=data.columns.droplevel(1)
    
    if data.isnull().values.any():
        missing_percentage = data.isnull().mean().sort_values(ascending=False)
        dropped_list = sorted(list(missing_percentage[missing_percentage > 0.005].index))
        data.drop(labels=dropped_list, axis=1, inplace=True)
        data = data.fillna(method='bfill', axis=0)
    
    data.to_csv('crypto_price_data_5m.csv')
    

def data_processor():
    if os.path.isfile('crypto_price_data_5m.csv'):
        with open('crypto_price_data_5m.csv', "r") as f:
            for line in f: pass
            # recent = line.strip().split(',')[0]
    else:
        new_data()
        
    data = pd.read_csv("crypto_price_data_5m.csv")
    data = data.set_index('Time')
    
    returns = data.pct_change()
    returns = returns.dropna()
    
    return data, returns

# =============================================================================
#     recent = dt.strptime(recent, '%Y-%m-%d %H:%M:%S')
#     time_minus = dt.now() - timedelta(days=3)
#     if (recent < time_minus):
#         data = datadict()
#         data = data.loc[:,data.columns.get_level_values(1).isin(['Close'])]
#         data.columns=data.columns.droplevel(1)
#         data.to_csv('crypto_price_data_5m.csv')
#     else:
# =============================================================================

for data