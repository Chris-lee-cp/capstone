# Copyright 2017 Junghoon Lee
#   jhoon.chris@gmail.com


import pandas_datareader as pdr
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
from pandas import Series, DataFrame, Panel
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


from sp_ml import SpMl


class SpLinearReg(SpMl):
    'stock prediction by linear regression'

    def __init__(self):
        pass
        
    def preprocess_fetures(self):
        #print self.ivv.head()
        nm_ivv = self.ivv.copy(deep=True)
        nm_ivv['Adj Close'] = nm_ivv['Adj Close']/nm_ivv['Adj Close'][0]
        nm_ivv['Volume'] = nm_ivv['Volume']/nm_ivv['Volume'].mean()
        nm_ivv['Open'] = (nm_ivv['Close']-nm_ivv['Open'])/nm_ivv['Open']
        nm_ivv['High'] = (nm_ivv['High']-nm_ivv['Low'])/nm_ivv['Low']
        nm_ivv = nm_ivv.rename(columns = {'Open':'Change'})
        nm_ivv = nm_ivv.rename(columns = {'High':'Maxofday'})
        #print nm_ivv.head()

        #print self.gld.head()
        nm_gld = self.gld.copy(deep=True)
        nm_gld['Adj Close'] = nm_gld['Adj Close']/nm_gld['Adj Close'][0]
        nm_gld['Volume'] = nm_gld['Volume']/nm_gld['Volume'].mean()
        #print nm_gld.head()

        self.prices = nm_ivv['Adj Close']
        self.features = nm_ivv.drop(['Close', 'Low'], axis = 1)

        # Rolling Mean 10 days
        rm10_ivv = pd.rolling_mean(nm_ivv['Adj Close'], window=10)
        rm10_ivv_ = pd.DataFrame(rm10_ivv)
        rm10_ivv_ = rm10_ivv_.rename(columns = {'Adj Close':'RM10'})

        # Rolling Mean 20 days
        rm20_ivv = pd.rolling_mean(nm_ivv['Adj Close'], window=20)
        rm20_ivv_ = pd.DataFrame(rm20_ivv)
        rm20_ivv_ = rm20_ivv_.rename(columns = {'Adj Close':'RM20'})
        #print rm20_ivv_.tail()

        # Rolling Mean 40 days
        rm40_ivv = pd.rolling_mean(nm_ivv['Adj Close'], window=40)
        rm40_ivv_ = pd.DataFrame(rm40_ivv)
        rm40_ivv_ = rm40_ivv_.rename(columns = {'Adj Close':'RM40'})

        #print gld.head()
        nm_gld = nm_gld.drop(['Open', 'High', 'Close', 'Low', 'Volume'], axis = 1)
        nm_gld = nm_gld.rename(columns = {'Adj Close':'GLD'})
        #print nm_gld.head()
        #print self.rm10_ivv_.head()
        #print "gold.dtypes", gold.dtypes

        self.features = self.features.join(nm_gld['GLD'], how='inner')
        self.features = self.features.join(rm10_ivv_, how='inner')
        self.features = self.features.join(rm20_ivv_, how='inner')
        self.features = self.features.join(rm40_ivv_, how='inner')
        print self.features.head()
        
        return

    
    def display_fetures(self):
        self.ivv[['Close','Adj Close', 'Open', 'High', 'Low']].plot(title='IVV price')
        plt.show()

        self.features['Volume'].plot(title='nomalized IVV Volume')
        plt.show()

        self.gld[['Close','Adj Close', 'Open', 'High', 'Low']].plot(title='GLD price')
        plt.show()

        self.features[['Adj Close', 'RM20', 'RM40']].plot(title='nomalized IVV price, 20-days, 40-days mean')
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

        print X_train
        #print X_train[:,1]
        print y_train

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        
        LR_result = pd.DataFrame(y_test)
        #print LR_result 
        #print y_pred
        #print y_test

        LR_result = LR_result.assign(Prediction = y_pred)
        # print LR_result
        LR_result = LR_result.rename(columns = {'Adj Close':'Real price'})
        LR_result[['Real price','Prediction']].plot(title='Linear Regression test result')
        plt.show()

        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print y_test, y_pred
        print("Mean squared error: %.2f" % mean_squared_error(y_test * self.ivv['Adj Close'][0], y_pred * self.ivv['Adj Close'][0]))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        
        return
    
    
    def predict_nextweek(self):
        # Predict future 1-week
        X_train = self.features[65:-5]
        y_train = self.prices[70:]
        X_future = self.features[-5:]

        regr = linear_model.LinearRegression()        
        regr.fit(X_train.values, y_train.values)

        # Make predictions using the testing set
        y_future = regr.predict(X_future)

        now = datetime.datetime.now()
        start_predict_day = now+datetime.timedelta(1)
        end_predict_day = now+datetime.timedelta(5)
        start_predict_day = start_predict_day.strftime("%Y-%m-%d")
        end_predict_day = end_predict_day.strftime("%Y-%m-%d")

        days = pd.date_range(start_predict_day, end_predict_day, freq='D')

        df = pd.DataFrame({'Date': days, 'Prediction': y_future})
        df = df.set_index('Date')
        df['Prediction'] = df['Prediction'] * self.ivv['Adj Close'][0]

        real_prices = self.prices[-10:] * self.ivv['Adj Close'][0]
        print real_prices
        print df 

        ax = real_prices.plot()
        df['Prediction'].plot(title='Linear Regression stock price forecasting', ax = ax)
        plt.show()
        return
    

