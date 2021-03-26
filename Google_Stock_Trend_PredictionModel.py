#LSTM_Project On Google Stock Prediction for Data ranging from 1-01-2005 to 31-12-2020
#Model Predicting the trend of Stock Price for Jan 2021

#Part1- Data Prerecessing-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset_train=pd.read_csv('GOOG_Stock_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

X_train=list()
y_train=list()
for i in range (70 ,4027):
    X_train.append(training_set_scaled[ i-70 :i ,0])
    y_train.append(training_set_scaled[ i,0])
X_train ,y_train =np.array(X_train) ,np.array(y_train)

X_train=np.reshape(X_train ,(X_train.shape[0], X_train.shape[1] , 1))

#Part2 -Building THE LSTM MODEL with 7 LSTM layers

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor= Sequential()

regressor.add(LSTM(units=80,return_sequences=True , input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80,return_sequences=True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units=80,return_sequences=False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer='adam' ,loss='mean_squared_error')
regressor.fit(X_train,y_train, batch_size=32 ,epochs=100)   

#Part3- Making The Actual Real Prediction for 2021 January

dataset_test=pd.read_csv('GOOG _Stock_Test.csv')
Actual_stock_Price= dataset_test.iloc[ :, 1:2].values

dataset_total= pd.concat((dataset_train['Open'] ,dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total) -len(dataset_test) -70 :].values
inputs=inputs.reshape(-1 ,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range (70,89):
    X_test.append(inputs[i-70:i ,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test ,(X_test.shape[0] ,X_test.shape[1], 1))
predicted_stock=sc.inverse_transform(regressor.predict(X_test))

#Visualizing The Results

plt.plot(Actual_stock_Price, color='red' ,label='Actual Stock Price of Jan-2021')
plt.plot(predicted_stock,color='blue' , label= 'Predicted Stock Price of Jan 2021')
plt.title('Google Stock Price Model')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()