# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:14:06 2019

@author: Gaurav Bothra
"""

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import os
#import telegram
#import shutil
import historical_data
import pandas as pd
import preprocess_data as ppd
import math
import lstm
import visualize as vs
import stock_data as sd
from keras.models import load_model

stock_list = {'HDFCBANK':'HDFCBANK', 'BAJAJ-AUTO':'BAJAJ-AUTO', 'HDFCLIFE':'HDFCLIFE', 'TCS':'TCS', 'RIIL':'RIIL', 'TATAPOWER':'TATAPOWER','INDIGO':'INDIGO', 'BPCL':'BPCL', 'BRITANNIA':'BRITANNIA', 'TATASTEEL':'TATASTEEL'}
#enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def error(bot, update, error):
    """Log Errors caused by Updates"""
    logger.warning('Update "%s" caused error "%s"', update, error)

def start(bot, update):
    """send a message when the command /start is issued."""
    reply = "Welcome to Stock Price Prediction.\nSend /help to see what i can do."
    update.message.reply_text(reply)
    
def get_stock_data(bot, update):
    message = update.message.text.split(" ")
    stock_code = stock_list[message[-1]]
    #if message[-1] == "IBM":
    #    stock_code = "EOD/IBM"
    first_time = check_first_time(message[-1])
    historical_data.get_stock_data(stock_code, first_time)
    update.message.reply_text("Stock Data Fetched.")

def get_news_data(bot, update):
    message = update.message.text.split(" ")
    stock_name = message[-1]
    #stock_code = stock_list[stock_name]
    #first_time = check_first_time(message[-1]+"_news")
    historical_data.get_news_data(stock_name)
    update.message.reply_text("News Data Fetched.")

def train(bot, update, retrain = False):
    message = update.message.text.split(" ")
    x_train, y_train, x_test, y_test, sc_close = preprocess(message[-1])
    model, prediction = train_model(message[-1], x_train, y_train, x_test, y_test, sc_close, retrain = retrain)
    accuracy = model_accuracy(model, x_train, y_train, x_test, y_test)
    update.message.reply_text(accuracy)

def predict_future(bot, update):
    message = update.message.text.split(" ")
    x_train, y_train, x_test, y_test, sc_close = preprocess(message[-1], test_data_size = 1)
    model, prediction = train_model(message[-1], x_train, y_train, x_test, y_test, sc_close)
    #accuracy = model_accuracy(model, x_train, y_train, x_test, y_test)
    p = []
    for i in prediction:
        #print(i)
        p = i
    #print(str(p[0]))
    update.message.reply_text("Next Closing price will be:"+str(p[0]))

def list_stocks(bot, update):
	l = ""
	for i in stock_list.keys():
		l+=i+"\n"
	update.message.reply_text(l)
	
def run(bot, update):
    get_stock_data(bot, update)
    get_news_data(bot, update)
    train(bot, update, retrain = True)
    predict_future(bot, update)

def help(bot, update):
    """send a message when the command /help is issued"""
    reply = '''send /run stock_name to fetch stock data.\n send /predict stock_name to predict next closing price.\n send /list to see available stocks.'''
    update.message.reply_text(reply)

def check_first_time(stock_name):
    if os.path.isfile(os.getcwd() + "/data/" + stock_name+".csv"):
        return False
    else:
        return True


def preprocess(stock_name, test_data_size = 200):
    path = os.getcwd() + "/data/" + stock_name + ".csv"
    data = pd.read_csv(path)
    data = data.dropna(how='any', axis=0)
    stocks = ppd.remove_data(data)
    #print(stocks.tail())
    stocks, sc_close = ppd.get_normalised_data(stocks)
    stocks_data = stocks.dropna(how='any', axis=0)
    unroll_length = 50
    #test_data_size = 200
    x_train, x_test, y_train, y_test = sd.train_test_split_lstm(stocks_data, prediction_time = 1, unroll_length = unroll_length, test_data_size=test_data_size)
    
    x_train = sd.unroll(x_train, unroll_length)
    x_test = sd.unroll(x_test, unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test = y_test[-x_test.shape[0]:]
    '''
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    '''
    return x_train, y_train, x_test, y_test, sc_close

def train_model(stock_name, x_train, y_train, x_test, y_test, sc_close ,unroll_length = 50, retrain = False):
    batch_size = 32
    epochs = 10
    if retrain:
        model = lstm.lstm_model(x_train.shape[-1], output_dim=unroll_length, return_sequences=True)
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        #start = time.time()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.05)
        model.save(os.getcwd() + "/data/" + stock_name+".h5")
        
    else:
        if not os.path.isfile(os.getcwd() + "/data/" + stock_name+".h5"):
            model = lstm.lstm_model(x_train.shape[-1], output_dim=unroll_length, return_sequences=True)
            
            model.compile(loss='mean_squared_error', optimizer='adam')
            #start = time.time()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.05)
            model.save(os.getcwd() + "/data/" + stock_name+".h5")
            #print('training_time', time.time() - start)
        else:
            model = load_model(os.getcwd() + "/data/" + stock_name+".h5")
    prediction = model.predict(x_test, batch_size=batch_size)
    prediction = sc_close.inverse_transform(prediction)
    #print(prediction)
    y_test = sc_close.inverse_transform(y_test)
    #vs.plot_lstm_prediction(y_test, prediction)
    return model, prediction

def model_accuracy(model, x_train, y_train, x_test, y_test):
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    accuracy = 'Train Score: %.8f MSE (%.8f RMSE)\n' % (trainScore, math.sqrt(trainScore))
    
    testScore = model.evaluate(x_test, y_test, verbose=0)
    accuracy += 'Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore))
    return accuracy


def main():
    TOKEN = "865487257:AAF4cZ3bFFfLndqeb-YMLmtMiMkbLwHP2jI"
    updater = Updater(TOKEN)
    
    dp = updater.dispatcher
    
    #commands - function
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("run", run))
    dp.add_handler(CommandHandler("get_stock_data", get_stock_data))
    dp.add_handler(CommandHandler("get_news_data", get_news_data))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("predict", predict_future))
    dp.add_handler(CommandHandler("list", list_stocks))
    dp.add_error_handler(error)
    #start the Bot
    updater.start_polling()
    
    updater.idle()
    
if __name__ == '__main__':
    main()