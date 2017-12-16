# Copyright 2017 Junghoon Lee
#   jhoon.chris@gmail.com

import pandas_datareader as pdr
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
from pandas import Series, DataFrame, Panel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from sp_ml import SpMl


class SpKnnReg(SpMl):
    'stock prediction by KNN regression'

    best_neighbor = 0
    week_forecast = pd.DataFrame()
    
    def __init__(self):
        pass
        
    def preprocess_fetures(self):
        #print self.ivv.head()
        nm_ivv = self.ivv.copy(deep=True)
        temp_ivv = self.ivv.copy(deep=True)
        
        temp_ivv = temp_ivv.shift(periods=1, freq=None, axis=0)
        #print nm_ivv.head()
        #print temp_ivv.head()
        
        nm_ivv['Adj Close'] = nm_ivv['Adj Close'] - temp_ivv['Adj Close']
        nm_ivv['Adj Close'][0] = 0
        nm_ivv['Adj Close'] = nm_ivv['Adj Close'] / self.ivv['Adj Close']

        #print "percentage value"
        #print self.nm_ivv['Adj Close']
        
        nm_ivv['Volume'] = nm_ivv['Volume']/nm_ivv['Volume'].mean()
        nm_ivv['Open'] = (nm_ivv['Close']-nm_ivv['Open'])/nm_ivv['Open']
        nm_ivv['High'] = (nm_ivv['High']-nm_ivv['Low'])/nm_ivv['Low']
        nm_ivv = nm_ivv.rename(columns = {'Open':'Change'})
        nm_ivv = nm_ivv.rename(columns = {'High':'Maxofday'})
        nm_ivv = nm_ivv.rename(columns = {'Adj Close':'Chg perc'})
        #print self.nm_ivv.head()

        #print self.gld.head()
        nm_gld = self.gld.copy(deep=True)
        nm_gld['Adj Close'] = nm_gld['Adj Close']/nm_gld['Adj Close'][0]
        nm_gld['Volume'] = nm_gld['Volume']/nm_gld['Volume'].mean()
        #print self.nm_gld.head()

        self.prices = nm_ivv['Chg perc']
        self.features = nm_ivv.drop(['Close', 'Low'], axis = 1)

        # Rolling Mean 20 days
        rm20_ivv = pd.rolling_mean(nm_ivv['Chg perc'], window=20)
        rm20_ivv_ = pd.DataFrame(rm20_ivv)
        rm20_ivv_ = rm20_ivv_.rename(columns = {'Chg perc':'RM20'})
        #print self.rm20_ivv_.tail()

        # Rolling Mean 40 days
        rm40_ivv = pd.rolling_mean(nm_ivv['Chg perc'], window=40)
        rm40_ivv_ = pd.DataFrame(rm40_ivv)
        rm40_ivv_ = rm40_ivv_.rename(columns = {'Chg perc':'RM40'})

        #print gld.head()
        nm_gld = nm_gld.drop(['Open', 'High', 'Close', 'Low', 'Volume'], axis = 1)
        nm_gld = nm_gld.rename(columns = {'Adj Close':'GLD'})
        #print gold.head()
        #print "gold.dtypes", gold.dtypes

        self.features = nm_ivv.drop(['Low', 'Close'], axis = 1)
        self.features = self.features.join(nm_gld, how='inner')
        self.features = self.features.join(rm20_ivv_, how='inner')
        self.features = self.features.join(rm40_ivv_, how='inner')

        print self.features.head()
        return

    
    def display_fetures(self):
        self.ivv[['Close','Adj Close', 'Open', 'High', 'Low']].plot(title='IVV price')
        plt.show()

        self.features['Volume'].plot(title='Nomalized IVV Volume')
        plt.show()

        self.gld[['Close','Adj Close', 'Open', 'High', 'Low']].plot(title='GLD price')
        plt.show()

        self.features['Chg perc'].plot(title='Nomalized IVV change percentage')
        plt.show()
        return

    
    def do_regression(self):
        X_train = self.features[65:-50]
        y_train = self.prices[70:-45]
        X_test = self.features[-50:-5]
        y_test = self.prices[-45:]

        # Show the results of the split
        print "Training set has {} samples.".format(X_train.shape[0])
        print "Testing set has {} samples.".format(X_test.shape[0])

        #print X_train.head()
        #print X_train[:,1]
        #print y_train.head()

        variance = -1000
        for count, n_neighbors in enumerate([5, 7, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80 ,90, 100, 200]):
            neigh = KNeighborsRegressor(n_neighbors, weights='distance')
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)
            print y_pred
            LR_result = pd.DataFrame(y_pred)
            LR_result.plot(title="n_neighbors %d" % n_neighbors)
            # The mean squared error
            print("n_neighbors : %d" % n_neighbors)
            print("Mean squared error: %.2f" % mean_squared_error(y_test*100, y_pred*100))
            # Explained variance score: 1 is perfect prediction
            score = r2_score(y_test, y_pred)
            if(variance < score):
                variance = score
                self.best_neighbor = n_neighbors

            print('Variance score: %.2f' % variance)
    
        plt.show()
        
        return
        
        
    def predict_nextweek(self):
        # Predict future 1-week
        X_train = self.features[65:-5]
        y_train = self.prices[70:]
        X_future = self.features[-5:]

        print "best neighbor = ", self.best_neighbor
        neigh = KNeighborsRegressor(n_neighbors=self.best_neighbor)
        neigh.fit(X_train, y_train)

        # Make predictions using the testing set
        y_future = neigh.predict(X_future)

        print y_future
        
        now = datetime.datetime.now()

        start_predict_day = now+datetime.timedelta(1)
        end_predict_day = now+datetime.timedelta(5)
        start_predict_day = start_predict_day.strftime("%Y-%m-%d")
        end_predict_day = end_predict_day.strftime("%Y-%m-%d")

        days = pd.date_range(start_predict_day, end_predict_day, freq='D')

        df = pd.DataFrame({'Date': days, 'Prediction': y_future})
        df = df.set_index('Date')
        
        
        df['Prediction'][0] = df['Prediction'][0] * self.ivv['Adj Close'][-1] + self.ivv['Adj Close'][-1]
        for i in range(1,5):
            print "i:", i
            df['Prediction'][i] = df['Prediction'][i] * df['Prediction'][i-1] + df['Prediction'][i-1]
        
        real_prices = self.ivv['Adj Close'][-10:]
        print real_prices            
        print df 

        self.week_forecast = df;

        ax = real_prices.plot()
        df['Prediction'].plot(title='KNN Regression stock price forecasting', ax = ax)
        plt.show()
        return
    
