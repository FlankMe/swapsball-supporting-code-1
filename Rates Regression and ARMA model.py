# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:49:23 2017 

This script downloads data from Bloomberg (note you will need a bloomberg
account connected first!) and performs a series of linear regressions 
and ARMA trainings. The results are plotted. 

@author: Riccardo Rossi
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
np.random.seed(int(time.time()))


from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima_model import ARMA


# Choose the key parameters
target = '.GB210 Index'
explanatory = [ 
            'USSW10 Index',
            '.US210 Index',
            'GBPUSD Curncy', 
            ]
start = '2013-01-01'
end = '2017-09-20'

# Load and pre-process the data
print 'Load and pre-process the data'
try:
    levels_df = (pd.DataFrame.from_csv('dataDump.csv').astype(float).loc[
                    pd.date_range(start, end, freq='B'), 
                    [target] + explanatory].dropna() 
                    )
except:
    """
    Download data and copy on file
    You will need to install TIA first, just type 'pip install tia' on cmd 
    prompt
    
    The idea is to download all the data you may ever need once, and then just
    load the file offline and select the data you need for the specific 
    analysis.
    """    
    import tia.bbg.datamgr as dm
    mgr = dm.BbgDataManager()
    securities = [ 
                'EUSA10 Index', 
                'USSW10 Index',
                'BPSW10 Index',
                'ASWABUND Index', 
                'ASWESHTZ Index',
                'ASWABOBL Index',
                'TYAISP Comdty', 
                '.EU210 Index',
                '.US210 Index',
                '.GB210 Index', 
                '.ITDE10 Index', 
                '.FRDE10 Index', 
                'SX5E Index', 
                'SPX Index',
                'UKX Index',
                'VIX Index', 
                'V2X Index', 
                'REFRDE Index', 
                'EONIA Index', 
                'EUSWEC Index', 
                'EUR003M Index', 
                'EURUSD Curncy', 
                'EURGBP Curncy',
                'JPY Curncy',
                ]
                
    levels_df = mgr[securities].get_historical('PX_LAST', 
                                               '1/1/2000', '09/20/2017')
    levels_df.to_csv('dataDump.csv')
    levels_df = (levels_df.astype(float).loc[
                    pd.date_range(start, end, freq='B'), 
                    [target] + explanatory].dropna() 
                    )
                    
# Pick the out-of-sample size
nPredictions = int(len(levels_df) * 0.10)

# Convert your data to stationary series
logDelta_df = np.log(levels_df).diff().dropna()

# Exclude outliers
fig = plt.figure(figsize=(6,4))
plt.hist(logDelta_df[target][:-nPredictions], bins=50)
plt.title("Histogram of " + target + "'s moves, in-sample only")
plt.show(fig)

percentile = max(abs(np.percentile(logDelta_df[target], 5)),
                 abs(np.percentile(logDelta_df[target], 95)))
for t in logDelta_df.index[:-nPredictions]:
    if np.abs(logDelta_df.loc[t][target]) > percentile:
        logDelta_df = logDelta_df.drop(t)
            
            
"""
Fit a (rolling) linear model
"""
print 'Run a linear regression model'

# Set the windown of rolling data
k0, k1, k2 = nPredictions*2, nPredictions, int(nPredictions/2)
K = [k0, k1, k2]

linearModel = linear_model.LinearRegression()
Betas = [pd.DataFrame(columns=explanatory + ['R^2']) for _ in range(len(K))]
for k in K:
    for t in range(k, len(logDelta_df) - nPredictions - 1):
        X = logDelta_df[t-k:t].drop(target, axis=1).copy()   
        y = logDelta_df[t-k:t][target].copy()   
        linearModel.fit(X.values, y.values)
        entry = np.reshape(np.array(
                    list(linearModel.coef_) + [linearModel.score(X, y)]),
                    (1, len(linearModel.coef_) + 1))
        Betas[K.index(k)] = Betas[K.index(k)].append(pd.DataFrame(data=entry, 
                                          columns=explanatory + ['R^2'], 
                                          index=[logDelta_df.index[t]]
                                          ))

for col in Betas[0].columns[:-1]:
    fig = plt.figure(figsize=(7,4))
    for i in range(len(K)):
        plt.plot(Betas[i][col], label='k = ' + str(K[i]))
    plt.title("k-rolling beta (on delta of log) for " + str(col))
    plt.legend()
    plt.show(fig)

   
"""
Test for significance of coefficients and drop insignificant variables
"""   
print 'Test for significance of coefficients'

X = logDelta_df[:-nPredictions].drop(target, axis=1).copy()   
y = logDelta_df[:-nPredictions][target].copy()   
SM_estimator = sm.OLS(y, sm.add_constant(X)).fit()
print(SM_estimator.summary())

# Remove non-explanatory variables
explanatory.remove('GBPUSD Curncy')
levels_df = levels_df.drop('GBPUSD Curncy', axis=1)
   
      
"""
Re-measure the betas on the differences and compare predicted levels 
to actual levels on the test set
"""
print 'Run the regression and compare the forecasts vs out-of-sample data'

delta_df = (levels_df).diff().dropna()
X = delta_df[-nPredictions-k:-nPredictions].drop(target, axis=1).copy()   
y = delta_df[-nPredictions-k:-nPredictions][target].copy()   
linearModel.fit(X.values, y.values)

XTest = delta_df[-nPredictions:].drop(target, axis=1).copy()   
yTest = delta_df[-nPredictions:][target].copy()      
y_at = linearModel.predict(XTest.values)

# Plot results
fig = plt.figure(figsize=(9,6))
plt.scatter(yTest.values, y_at)  
plt.plot(np.arange(yTest.values.min(), yTest.values.max()), 
         np.arange(yTest.values.min(), yTest.values.max()), 'r')
plt.title("Scatter plot of predicted moves vs actual moves (in bps)")
plt.xlabel("Actual moves")
plt.ylabel("Predicted moves")
plt.show(fig)

fig = plt.figure(figsize=(6,4))
residuals = y_at - yTest.values
stats.probplot(residuals, dist='norm', plot=plt)
plt.title("Normal QQ plot of residuals")
plt.show(fig)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residuals, 
                               lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residuals, 
                                lags=40, ax=ax2)
           
fig = plt.figure(figsize=(6,4))
plt.hist(residuals)              
plt.title("Histogram of the residuals")
plt.show(fig)

fig = plt.figure(figsize=(6,6))
plt.scatter(yTest.values, yTest.values - y_at)  
plt.plot(np.arange(yTest.values.min(), yTest.values.max()), 
         np.zeros(len(np.arange(yTest.values.min(), yTest.values.max()))), 'r')
plt.title("Scatter plot of residuals vs actual moves (in bps)")
plt.xlabel("Actual moves")
plt.ylabel("Residuals")
plt.show(fig)

print "Latest Betas:\n", explanatory, '\n', linearModel.coef_
print "R^2 coefficient = ", linearModel.score(XTest, yTest)


"""
Analysis of a mean-reverting process
"""
print 'Attempt to construct a mean-reverting process'

# Load and pre-process the data
target = 'ASWABOBL Index'
explanatory = [ 
           'ASWABUND Index', 
           'ASWESHTZ Index', 
            ]
start = '2013-01-01'
end = '2015-01-01'
levels_df = (pd.DataFrame.from_csv('dataDump.csv').astype(float).loc[
                    pd.date_range(start, end, freq='B'), 
                    [target] + explanatory].dropna() 
                    )

# Pick the out-of-sample size and exclude outliers
nPredictions = k = int(len(levels_df) * 0.25)
delta_df = (levels_df).diff().dropna()

percentile = max(abs(np.percentile(delta_df[target], 5)),
                 abs(np.percentile(delta_df[target], 95)))
for t in delta_df.index[:-nPredictions]:
    if np.abs(delta_df.loc[t][target]) > percentile:
        levels_df = levels_df.drop(delta_df.loc[t].name)

# Run a regression
X = delta_df[-nPredictions-k:-nPredictions].drop(target, axis=1).copy()   
y = delta_df[-nPredictions-k:-nPredictions][target].copy()   
linearModel.fit(X.values, y.values)

# Construct a stationary time series and remove outliers
Betas = pd.DataFrame(data=np.reshape(linearModel.coef_, 
                                    (1, len(linearModel.coef_))), 
                     columns=explanatory)
betaAdj = levels_df[levels_df.drop(target, axis=1).columns] * Betas.mean()

statSeries = pd.DataFrame(data=levels_df[target].values - betaAdj.sum(axis=1),
                          index=levels_df.index,
                          columns=['Stationary series']
                          )
                          
# Plot the history of the stationary series          
fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twinx()
ax1.plot(statSeries, label='Stationary', color='b')
ax2.plot(levels_df[target], label=str(target), color='r')
plt.title("History of (hypothetically) stationary series vs target series")
ax1.set_ylabel('Stationary', color='b')
ax2.set_ylabel(str(target), color='r')
plt.show(fig)


# Parameters of the AR-MA model
print 'Running the ARMA model'

freq, nSeries = 'B', 1
lagOrder = 4
p, q = lagOrder, lagOrder
window = 12 if freq == 'M' else 120

# Calibrate and plot
try:
    predValues = []
    for j in range(nSeries):
        arma = ARMA(statSeries[(nSeries-j)*k:-nPredictions-j*k], 
                               order=(p, q), freq=freq
                               ).fit()
        predValues.append(pd.DataFrame(arma.predict(
            start=len(statSeries[(nSeries-j)*k:-nPredictions-j*k]), 
            end=len(statSeries[(nSeries-j)*k:-nPredictions-j*k]) + window,
            dynamic=True)))
            
    fig = plt.figure(figsize=(9,6))
    plt.plot(statSeries, label='Stationary')
    for j in range(nSeries):
        plt.plot(predValues[j], label='Prediction')
    plt.title("History of the series and predicted values")
    plt.legend()
    plt.show(fig)
        
except:
    print 'Failed to calibrate the ARMA model'
                 
