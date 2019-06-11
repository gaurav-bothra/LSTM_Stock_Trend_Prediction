# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:05:15 2019

@author: Gaurav Bothra
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_normalised_data(data):
    '''
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume', 'sentiment']
    data[numerical] = scaler.fit_transform(data[numerical])
    '''
    sc_open = MinMaxScaler()
    data['Open'] = sc_open.fit_transform(data['Open'])
    sc_close = MinMaxScaler()
    data['Close'] = sc_close.fit_transform(data['Close'])
    sc_volume = MinMaxScaler()
    data['Volume'] = sc_volume.fit_transform(data['Volume'])
    sc_senti = MinMaxScaler()
    data['sentiment'] = sc_senti.fit_transform(data['sentiment'])
    #print(data)
    return data, sc_close

def remove_data(data):
    item = []
    open = []
    close = []
    volume = []
    sentiment = []
    
    i_counter = 0
    for i in range(len(data)-1, -1, -1):
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        volume.append(data['Volume'][i])
        sentiment.append(data['sentiment'][i])
        i_counter += 1
    
    stocks = pd.DataFrame()
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = close
    stocks['Volume'] = volume
    stocks['sentiment'] = sentiment
    return stocks
    