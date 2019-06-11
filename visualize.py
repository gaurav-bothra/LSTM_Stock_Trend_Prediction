# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:17:03 2019

@author: Gaurav Bothra
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18, 12)


def price(x):
    return '$%1.2f' % x

def plot_lstm_prediction(message, actual, prediction, title='Actual vs Prediction', y_label='Price USD', x_label='Trading Days'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.savefig('../static/img/'+message+'result.png')
    #plt.show()
    
