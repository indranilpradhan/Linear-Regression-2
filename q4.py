import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import datetime as dt

class Weather:
    def __init__(self, learning_rate = 0.001,epochs = 100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def fit_test(self, X_test):
        self.X_test = X_test
    
    def fit_coef(self, coef):
        self.coef = coef
        
    def prediction(self, X_test, newcoeff):
        return self.X_test.dot(newcoeff)
    
    
    
    def cost_calculation(self, X_train, Y_train, Coefficient):
        leng = len(Y_train)
        sumcost = np.sum((self.X_train.dot(Coefficient) - self.Y_train) ** 2)
        cost = sumcost/(2 * leng)
        return cost
    
    def calculate_gradient(self, weights,X_train, Y_train):
        leng = len(self.Y_train)
        return self.X_train.T.dot(weights-self.Y_train) / leng
    
    def calculate_coefficients(self, coefficients, learning_rate, gradient):
        cal_grad = learning_rate * gradient
        penalty = coefficients - cal_grad
        return penalty
    
    def gradient_descent(self, X_train, Y_train, coefficients, learning_rate, epochs):
        costs = []
        for i in range(epochs):
            weights = self.X_train.dot(coefficients)
            gradient = self.calculate_gradient(weights, X_train, Y_train)
            coefficients = self.calculate_coefficients(coefficients, learning_rate, gradient)
            cost = self.cost_calculation(X_train, Y_train, coefficients)
            costs.append(cost)
        return coefficients, costs
    
    
    
    def train(self, path):
        df = pd.read_csv(str(path))
        enc = LabelEncoder()
        df['Precip Type'] = enc.fit_transform(df['Precip Type'].astype('str'))
        df['Summary'] = enc.fit_transform(df['Summary'].astype('str'))
        df['Daily Summary'] = enc.fit_transform(df['Daily Summary'].astype('str'))
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])
        df['Formatted Date'] = df['Formatted Date'].map(dt.datetime.toordinal)
        df['Formatted Date']=(df['Formatted Date']-df['Formatted Date'].min())/(df['Formatted Date'].max()-df['Formatted Date'].min())
        df['Summary']=(df['Summary']-df['Summary'].min())/(df['Summary'].max()-df['Summary'].min())
        df['Temperature (C)']=(df['Temperature (C)']-df['Temperature (C)'].min())/(df['Temperature (C)'].max()-df['Temperature (C)'].min())
        df['Wind Speed (km/h)']=(df['Wind Speed (km/h)']-df['Wind Speed (km/h)'].min())/(df['Wind Speed (km/h)'].max()-df['Wind Speed (km/h)'].min())
        df['Wind Bearing (degrees)']=(df['Wind Bearing (degrees)']-df['Wind Bearing (degrees)'].min())/(df['Wind Bearing (degrees)'].max()-df['Wind Bearing (degrees)'].min())
        df['Visibility (km)']=(df['Visibility (km)']-df['Visibility (km)'].min())/(df['Visibility (km)'].max()-df['Visibility (km)'].min())
        df['Pressure (millibars)']=(df['Pressure (millibars)']-df['Pressure (millibars)'].min())/(df['Pressure (millibars)'].max()-df['Pressure (millibars)'].min())
        df['Daily Summary']=(df['Daily Summary']-df['Daily Summary'].min())/(df['Daily Summary'].max()-df['Daily Summary'].min())

        train = df
        Y_train_temp = train['Apparent Temperature (C)'].copy()
        X_train = train.drop('Apparent Temperature (C)', 1)
        X_train = np.array(X_train).astype('float')
        Y_train = np.array(Y_train_temp)
        
        self.fit(X_train,Y_train)
        coefficients = np.full(self.X_train.shape[1],1.4)

        newcoeff, cost = self.gradient_descent(self.X_train, self.Y_train, coefficients, self.learning_rate, self.epochs)
        self.fit_coef(newcoeff)
        
    def predict(self, path):
        df = pd.read_csv(str(path))
        enc = LabelEncoder()
        df['Precip Type'] = enc.fit_transform(df['Precip Type'].astype('str'))
        df['Summary'] = enc.fit_transform(df['Summary'].astype('str'))
        df['Daily Summary'] = enc.fit_transform(df['Daily Summary'].astype('str'))
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])
        df['Formatted Date'] = df['Formatted Date'].map(dt.datetime.toordinal)
        df['Formatted Date']=(df['Formatted Date']-df['Formatted Date'].min())/(df['Formatted Date'].max()-df['Formatted Date'].min())
        df['Summary']=(df['Summary']-df['Summary'].min())/(df['Summary'].max()-df['Summary'].min())
        df['Temperature (C)']=(df['Temperature (C)']-df['Temperature (C)'].min())/(df['Temperature (C)'].max()-df['Temperature (C)'].min())
        df['Wind Speed (km/h)']=(df['Wind Speed (km/h)']-df['Wind Speed (km/h)'].min())/(df['Wind Speed (km/h)'].max()-df['Wind Speed (km/h)'].min())
        df['Wind Bearing (degrees)']=(df['Wind Bearing (degrees)']-df['Wind Bearing (degrees)'].min())/(df['Wind Bearing (degrees)'].max()-df['Wind Bearing (degrees)'].min())
        df['Visibility (km)']=(df['Visibility (km)']-df['Visibility (km)'].min())/(df['Visibility (km)'].max()-df['Visibility (km)'].min())
        df['Pressure (millibars)']=(df['Pressure (millibars)']-df['Pressure (millibars)'].min())/(df['Pressure (millibars)'].max()-df['Pressure (millibars)'].min())
        df['Daily Summary']=(df['Daily Summary']-df['Daily Summary'].min())/(df['Daily Summary'].max()-df['Daily Summary'].min())    
    
        train = df
        X_test = train
        X_test = np.array(X_test).astype('float')

        self.fit_test(X_test)
        Y_predict = self.prediction(self.X_test, self.coef)
        return Y_predict
        
        
        