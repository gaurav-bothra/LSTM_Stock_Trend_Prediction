# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:42:44 2019

@author: Gaurav Bothra
"""

#import flask
from flask import Flask, request, render_template
from bot import preprocess, train_model, check_first_time, model_accuracy
import historical_data
import visualize as vs
app = Flask(__name__,template_folder='templates')

stock_list = {'HDFCBANK':'HDFCBANK', 'BAJAJ-AUTO':'BAJAJ-AUTO', 'HDFCLIFE':'HDFCLIFE', 'TCS':'TCS', 'RIIL':'RIIL', 'TATAPOWER':'TATAPOWER','INDIGO':'INDIGO', 'BPCL':'BPCL', 'BRITANNIA':'BRITANNIA', 'TATASTEEL':'TATASTEEL'}

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', stock_list=stock_list, img="HDFCBANK")

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    message = request.form['stock_name']
    first_time = check_first_time(message)
    print(stock_list[message])
    historical_data.get_stock_data("EOD/IBM", first_time)
    label = "Stock Data Fetched."
    return render_template('index.html', label=label, stock_list=stock_list)

@app.route('/get_news_data', methods=['POST'])
def get_news_data():
    stock_name = request.form['stock_name']
    #first_time = check_first_time(message[-1]+"_news")
    historical_data.get_news_data("EOD/IBM")
    label = "News Data Fetched."
    return render_template('index.html', label=label, stock_list=stock_list)

@app.route('/train', methods=['POST'])
def train(retrain = True):
    message = "IBM"
    x_train, y_train, x_test, y_test, sc_close = preprocess(message)
    model, prediction = train_model(message, x_train, y_train, x_test, y_test, sc_close, retrain = retrain)
    accuracy = model_accuracy(model, x_train, y_train, x_test, y_test)
    label = accuracy
    vs.plot_lstm_prediction(message, y_test, prediction)
    return render_template('index.html', label=label, stock_list=stock_list, img=message)

@app.route('/predict', methods=['POST'])
def predict_future():
    message = "IBM"
    x_train, y_train, x_test, y_test, sc_close = preprocess(message, test_data_size = 1)
    model, prediction = train_model(message, x_train, y_train, x_test, y_test, sc_close)
    #accuracy = model_accuracy(model, x_train, y_train, x_test, y_test)
    p = []
    for i in prediction:
        #print(i)
        p = i
    #print(str(p[0]))
    #update.message.reply_text("Tommorow's price will be:"+str(p[0]))
    label = "Next Closing price will be:"+str(p[0])
    return render_template('index.html', label=label, stock_list=stock_list)

if __name__ == '__main__':
    app.run()