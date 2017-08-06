import random

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import Adam

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from com.kailin.api_image import api_image
from com.kailin.api_file import api_file
from matplotlib import pyplot

import numpy

stock2330 = api_file.readExcel('E:/pythonwork/data/2330.xlsx')
stock2330_Y = stock2330['收盤'].values
stock2330_X = stock2330['日期'].values.astype('float32')

msk = len(stock2330) - 120
train_Y = stock2330_Y[:msk]
train_X = stock2330_X[:msk]
test_Y = stock2330_Y[msk:]
test_X = stock2330_X[msk:]

print(test_X[:10])

numpy.random.seed(10)


model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, activation="softmax"))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(numpy.array(train_X), numpy.array(train_Y), batch_size=120, validation_split=0.1, epochs=1000, verbose=2)
api_image.showTrainHistory(history, 'loss', 'val_loss')


trainScore = model.evaluate(train_X, train_Y, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(test_X, test_Y, verbose=0)
print('Test Score: ', testScore)

# model = Sequential()
# # build a LSTM RNN
# model.add(LSTM(
#     batch_input_shape=(10, 5865, 10),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
#     output_dim=10,
#     return_sequences=True,      # True: output at all steps. False: output as last step. 對於每一個時間點，是否要輸出 output
#     stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2 Batch 之間狀態是否有聯繫
# ))
# model.add(TimeDistributed(Dense(1)))
# model.compile(optimizer='adam',loss='mse')
# history = model.fit(train_data, train_label, batch_size=40, validation_split=0.2, epochs=1000, verbose=2)

#
# model = Sequential()
# model.add(LSTM(batch_input_shape=(404, 1, 404)))
# model.add(Dense(units=1, activation='relu', input_dim=1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(train_data, train_label, batch_size=40, validation_split=0.2, epochs=1000, verbose=2)
# api_image.showTrainHistory(history, 'loss', 'val_loss')
#
# cost = model.evaluate(test_data, test_label, verbose=1)
# print('\n', cost, '\n')
# predict = model.predict(test_data)
# for j in range(0, 102):
#     print('No.', j + 1, ' price : ', test_label[j], 'predict price : ', predict[j])
