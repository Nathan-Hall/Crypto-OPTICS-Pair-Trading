import pandas as pd
from binance.client import Client
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import datetime as dt
from datetime import timedelta
import requests
import concurrent.futures
import csv
from process_data import data_processor

def splitdata(ts):
    ts_n = len(ts)
    train_n = int(ts_n * 0.8)
    train_data = ts.iloc[:train_n]
    test_data = ts.iloc[train_n:ts_n]
    
    return train_data, test_data


def findpairs(data_train, data_test):
    """
     tradeable pairs need to verify cointegration, min half life, have a min
     no. of zero crossings
     tests:
         cointegration p val
         half life >78, <20000
         hurst threshhold = 0.5 (mean reversion test)
     return pairs that pass tests
    """
    n = data_train.shape[1]
    keys = data_train.keys()
    pairs_fail_criteria = {'cointegration': 0, 'hurst_exponent': 0, 
                            'half_life': 0, 'mean_cross': 0, 'None': 0}
    pairs = []
    
    for i in range(n):
         for j in range(i+1, n):
            S1_train = data_train[keys[i]]; S2_train = data_train[keys[j]]
            S1_test = data_test[keys[i]]; S2_test = data_test[keys[j]] 
            result, criteria_not_verified = check_properties((S1_train, S2_train), (S1_test, S2_test))
            
            pairs_fail_criteria[criteria_not_verified] += 1
            if result is not None:
                pairs.append((keys[i], keys[j], result))
                
    return pairs, pairs_fail_criteria


def check_properties(train_series, test_series, p_val_threshhold=0.1, 
                     p_val_pair=0.05, min_half_life=20, max_half_life=20000, 
                     min_zero_crossings=0, hurst_threshold=0.5, subsample=0):
    X = train_series[0]
    Y = train_series[1]
    pairs = [(X, Y), (Y, X)]
    pair_stats = [0] * 2
    
    #Test for integration
    y_pval = adfuller(Y)[1]
    criteria_not_verified = 'cointegration'
    if y_pval > p_val_threshhold:
        x_pval = adfuller(X)[1]
        if x_pval > p_val_threshhold:
            #cointegrate
            for i, pair in enumerate(pairs):
                S1 = np.asarray(pair[0])
                S2 = np.asarray(pair[1])
                S1_c = sm.add_constant(S1)

                # Y = bX + c
                # ols: (Y, X)
                results = sm.OLS(S2, S1_c).fit()
                b = results.params[1]

                if b > 0:
                    spread = pair[1] - b * pair[0] # as Pandas Series
                    spread_array = np.asarray(spread) # as array for faster computations
                    p_val = adfuller(spread_array)[1]
                    t_stat = adfuller(spread_array)[0]
                    if p_val > p_val_pair:
                        criteria_not_verified = 'hurst_exponent'
                        #hurst
                        hurst_exp = hurst(spread_array)
                        if hurst_exp < hurst_threshold:
                            criteria_not_verified = 'half_life'
                            #half life test
                            hl = halflife(spread_array)
                            if (hl >= min_half_life) and (hl < max_half_life):
                                criteria_not_verified = 'mean_cross'
                                zero_cross = zero_crossings(spread_array)
                                if zero_cross >= min_zero_crossings:
                                    criteria_not_verified = 'None'
                                    pair_stats[i] = {
                                              't_statistic': t_stat,
                                              'p_value': p_val_pair,
                                              'coint_coef': b,
                                              'zero_cross': zero_cross,
                                              'half_life': int(round(hl)),
                                              'hurst_exponent': hurst_exp,
                                              'spread': spread,
                                              'Y_train': pair[1],
                                              'X_train': pair[0]
                                              }
    if pair_stats[0] == 0 and pair_stats[1] == 0:
        result = None
        return result, criteria_not_verified
    
    elif pair_stats[0] == 0:
        result = 1
    elif pair_stats[1] == 0:
        result = 0
    else: # both combinations are possible
        # select lowest t-statistic as representative test
        if abs(pair_stats[0]['t_statistic']) > abs(pair_stats[1]['t_statistic']):
            result = 0
        else:
            result = 1
    
    if result == 0:
        result = pair_stats[0]
        result['X_test'] = test_series[0]
        result['Y_test'] = test_series[1]
    elif result == 1:
        result = pair_stats[1]
        result['X_test'] = test_series[1]
        result['Y_test'] = test_series[0]
    
    return result, criteria_not_verified
    

def pairs_overlap(pairs, p_value_threshold, min_zero_crossings, min_half_life, hurst_threshold):
    
    pairs_overlapped = []
    pairs_overlapped_index = []

    for index, pair in enumerate(pairs):
        # get consituents
        X = pair[2]['X_test']
        Y = pair[2]['Y_test']
        # check if pairs is valid
        series_name = X.name
        X = sm.add_constant(X)
        results = sm.OLS(Y, X).fit()
        X = X[series_name]
        b = results.params[X.name]
        spread = Y - b * X
        stats = self.check_for_stationarity(pd.Series(spread, name='Spread'))

        if stats['p_value'] < p_value_threshold:  # verifies required pvalue
            hl = self.calculate_half_life(spread)
            if hl >= min_half_life:  # verifies required half life
                zero_cross = self.zero_crossings(spread)
                if zero_cross >= min_zero_crossings:  # verifies required zero crossings
                    hurst_exponent = self.hurst(spread)
                    if hurst_exponent < hurst_threshold:  # verifies hurst exponent
                        pairs_overlapped.append(pair)
                        pairs_overlapped_index.append(index)

    return pairs_overlapped, pairs_overlapped_index


def hurst(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0


def halflife(z_array):
    z_lag = np.roll(z_array, 1)
    z_lag[0] = 0
    z_ret = z_array - z_lag
    z_ret[0] = 0

    # adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)

    model = sm.OLS(z_ret[1:], z_lag2[1:])
    res = model.fit()

    halflife = -np.log(2) / res.params[1]

    return halflife


def zero_crossings(x):
    """
    Function that counts the number of zero crossings of a given signal
    :param x: the signal to be analyzed
    """
    x = x - x.mean()
    zero_crossings = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x[i] * x[i + 1] < 0) or (x[i] == 0)))

    return zero_crossings


def find_clusters(returns, prices):
    prin_components = 5
    pca = PCA(n_components=prin_components)
    pca.fit(returns)
    
    normed = np.hstack(pca.components_.T)
    normed = preprocessing.StandardScaler().fit_transform(pca.components_.T)
    
    clf = OPTICS(min_samples=3, max_eps=2, xi=0.05, metric='euclidean', cluster_method='xi')
    clf.fit(normed)
    
    clustered = clf.labels_
    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    #clustered_series_all = pd.Series(index=ret.columns, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    
    plot_clusters(clustered_series, prices)
    
    return clustered_series


def plot_clusters(clustered_series, prices):
    counts = clustered_series.value_counts()
    cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]
    
    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 3)]:
        tickers = list(clustered_series[clustered_series==clust].index)
        means = np.log(prices[tickers].mean())
        data = np.log(prices[tickers]).sub(means)
        data.plot(title='Time Series for Cluster %d' % clust)
    
    
def get_possible_pairs(clustered_series, train, test):
    total_pairs, total_pairs_fail_criteria = [], []
    n_clusters = len(clustered_series.value_counts())
    symbols = []
    
    for clust in range(n_clusters):
        symbols = list(clustered_series[clustered_series == clust].index)
        cluster_train = train[symbols]
        cluster_test = test[symbols]
        pairs, pairs_fail_criteria = findpairs(cluster_train, cluster_test)
        
        total_pairs.extend(pairs)
        total_pairs_fail_criteria.append(pairs_fail_criteria)
    
    unique_tickers = np.unique([(element[0], element[1]) for element in total_pairs])
    
    return total_pairs, unique_tickers


def pairs_from_csv():
    pairs = pd.read_csv('pairs_found.csv', header=None)


def main():
    closesdf, ret = data_processor()
    train, test = splitdata(closesdf)
    clustered_ts = find_clusters(ret, closesdf)
    pairs, unique_pairs = get_possible_pairs(clustered_ts, train, test)
    with open('pairs_found.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(pairs)
    
    return pairs, closesdf


if __name__ == "__main__":
    all_pairs, data = main()

    


