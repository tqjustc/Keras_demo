
import pdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(units=32, return_sequences=True))
model.add(LSTM(units=32))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )

x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

