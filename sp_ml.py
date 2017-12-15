# Copyright 2017 Junghoon Lee
#   jhoon.chris@gmail.com


import pandas_datareader as pdr
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
from pandas import Series, DataFrame, Panel

class SpMl:
    'stock prediction class'
    
    ivv = pd.DataFrame()
    gld = pd.DataFrame()
    prices = pd.DataFrame()
    features = pd.DataFrame()

    def __init__(self):
        pass
    
    
    def get_input_feature(self, ticker, start_date, end_date):
        print "loading : ", ticker, "from : ", start_date, " to : ", end_date
        return pdr.get_data_yahoo(ticker,start_date, end_date)
    
    
    def get_input_feature_from_file(self, file):
        print "loading : ", file
        return pd.read_csv(file)


