# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:14:06 2019

@author: Gaurav Bothra
"""
import os
import quandl
quandl.ApiConfig.api_key = "LZeiFF8D_tvy6V2aj-Ca"
import tweepy as tp
from datetime import date, timedelta
from textblob import TextBlob
import pandas as pd
from random import randrange
def get_stock_data(stock_name, first_time=False):
    stock_name = "EOD/IBM"
    print(stock_name)
    #end_date = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    if first_time:
        start_date = (date.today() - timedelta(60*30)).strftime('%Y-%m-%d')
        #start_date = (date.today() - timedelta(6*30)).strftime('%Y-%m-%d')
    else:
        start_date = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    data = quandl.get(stock_name, start_date=start_date)
    data['sentiment'] = 0
    for ind, row in data.iterrows():
        data.at[ind, 'sentiment'] = randrange(-10, 10)
    save = stock_name.split('/')[-1] + ".csv"
    path = os.getcwd() + "/data/" + save
    if first_time:
        data.to_csv(path)
    else:
        data.to_csv(path, mode='a', header=False)
    
def get_news_data(stock_name,first_time=False):
    consumer_key = 'buIbsIcCmFz4TtjMZhkle5yj9'
    consumer_secret = 'UGkwtPUgeZHsYMtnFoMUfhkSGQPqGPKpIvurwxRVucIeCadXjG'
    access_token = '2740611399-EG4rVgPrIduzCi2FUgTnRYuP5hAXITRA6NV0eTv'
    access_token_secret = 'YsbFiqYxL1LH9gVEtpHyoRMf7hDfbUlNQW8V8qNp1h84o'
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tp.API(auth)
    search_words = stock_name + " OR finance -filter:retweets"
    #print(search_words)
    
    if first_time:
        since = (date.today() - timedelta(6)).strftime('%Y-%m-%d')
    else:
        since = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    tweets = tp.Cursor(api.search, q= search_words, lang='en', since = since).items(2000)
    #for t in tweets:
    #    print(t.created_at.strftime('%Y-%m-%d'), t.text)
    save = stock_name.split('/')[-1] + ".csv"
    path = os.getcwd() + "/data/" + save
    data = pd.read_csv(path)
    senti = data.loc[data['Date'] == since]['sentiment']
    #print(since)
    #f = csv.writer(open(path, 'a', newline=''))
    try:
        for t in tweets:
            blob = TextBlob(t.text)
            senti += blob.sentiment.polarity
        #f.writerow((t.created_at.strftime('%Y-%m-%d'), senti))
        data.at[data['Date'] == since, 'sentiment'] = senti
        data.to_csv(path)
    except Exception as e:
        print(e)
# get_news_data("BAJAJ AUTO", True)
#get_stock_data("EOD/IBM", first_time=True)
#get_news_data("IBM", first_time=True)
