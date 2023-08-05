import keras
import numpy as np
import pandas as pd
import pandas_datareader.data as wb
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st


import datetime as dt
import os
start = dt.datetime(2010,1,1)
end =dt.datetime(2023,5,25)
import os
from pandas_datareader import  data as pdr
import yfinance as yf
yf.pdr_override()


st.title('Stock trend Prediction')
user_input = st.text_input('Enter stock symbol','TSLA')
data = pdr.get_data_yahoo(user_input,start,end)

st.subheader('Stock data from 2010-2023')
st.write(data.describe())
# to run py D:/PotatoDeduction/PlantVillage/Training/stockPrepiction> -m streamlit run app.py

st.subheader('Close price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Close price vs Time chart with 100MvA')
mav100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(mav100)
st.pyplot(fig)

st.subheader('Close price vs Time chart with 200MvA and 100MVA')
mav100 = data.Close.rolling(100).mean()
mav200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(mav100)
plt.plot(mav200)
st.pyplot(fig)

data_training = pd.DataFrame(data['Close'] [0:int(len(data) * 0.70)])
# after 70 percentage to end
data_testing  = pd.DataFrame(data['Close'][int(len(data) * 0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = Scaler.fit_transform(data_training)


model = load_model('keras_model.h5')

past100_days = data_training.tail(100)

#final_data = past100_days.append(data_testing,ignore_index=True)
final_data = pd.concat([past100_days,data_testing],axis=0,ignore_index=True)

inputdata = Scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100,inputdata.shape[0]):
    x_test.append(inputdata[i-100:i])
    y_test.append(inputdata[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)


y_predicted = model.predict(x_test)


scaler = Scaler.scale_

scaler_factor = 1 / scaler[0]

y_predicted = y_predicted * scaler_factor
y_test  = y_test * scaler_factor

st.subheader('Final Predicted chart')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original Price')
plt.plot(y_predicted, 'r', label ='Predited Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
