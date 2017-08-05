import random

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from com.kailin.api_image import api_image
from com.kailin.api_file import api_file

from matplotlib import pyplot

import numpy

(train_data, train_label), (test_data, test_label) = boston_housing.load_data()

linreg = LinearRegression()
linreg.fit(train_data, train_label)
train_data_f = linreg.predict(train_data)
test_data_f = linreg.predict(test_data)


# 計算MES、RMSE
print("train MSE:", metrics.mean_squared_error(test_label, test_data_f))
print("train RMSE:", numpy.sqrt(metrics.mean_squared_error(test_label, test_data_f)))
# 繪製曲線圖
# pyplot.scatter(train_data_f, train_label)
# pyplot.show()
# pyplot.scatter(test_data_f, test_label)
# pyplot.show()

numpy.random.seed(10)

model = Sequential()
model.add(Dense(units=1, activation='relu', input_dim=1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_data_f, train_label, batch_size=40, validation_split=0.2, epochs=1000, verbose=2)
api_image.showTrainHistory(history, 'loss', 'val_loss')

cost = model.evaluate(test_data_f, test_label, verbose=1)
print('\n', cost, '\n')
predict = model.predict(test_data_f)
for j in range(0, 102):
    print('No.', j+1, ' price : ', test_label[j], 'predict price : ', predict[j])

