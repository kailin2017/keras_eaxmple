from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

from com.kailin.api_image import api_image
from com.kailin.api_file import api_file
from matplotlib import pyplot

import numpy

numpy.random.seed(10)

clos = ['日期', '收盤', '開盤', '最高', '最低', '成交量', '成交金額']
# clos = ['日期', '收盤', '開盤']
clos_count = len(clos) - 1


def timestarp2string(dataset):
    result = []
    for i in range(len(dataset['日期'])):
        result.append(str(dataset['日期'][i])[:10].replace('-', ''))
    dataset['日期'] = result
    return dataset


def createDataset(dataset, look_back=1):
    dataX = []
    dataY = []
    for i in range(len(dataset) - look_back - 1 - 5):
        tempX = []
        for j in range(1, clos_count + 1):
            tempX.append(dataset[i:(i + look_back), j])
        dataX.append(tempX)
        dataY.append(dataset[i + look_back + 5, 1])
    return numpy.array(dataX), numpy.array(dataY)


def futureDataset(data):
    size = len(data)
    dataX = []
    dataX.append(data[size - 2:size - 1])
    return numpy.array(dataX)


stockcode = '2409'
pathXlsx = api_file.dataPath + stockcode + '.xlsx'
pathh5 = api_file.dataPath + stockcode + 'lstm.h5'

stock2330 = api_file.readExcel(pathXlsx)
stock2330_X = stock2330['日期'].values
stock2330_Y = stock2330['收盤'].values
stock2330_XY = timestarp2string(stock2330)[clos].values.astype('float64')

look_back = 30
look_predict = len(stock2330_XY) - 60
train, test = stock2330_XY[:look_predict], stock2330_XY[look_predict:len(stock2330_XY)]
trainX, trainY = createDataset(train, look_back)
testX, testY = createDataset(test, look_back)
futureX, futureY = createDataset(stock2330_XY, look_back)

model = Sequential()
model.add(LSTM(1024, dropout=0.2, input_shape=(clos_count, look_back)))
# model.add(LSTM(lstmoutputdim, return_sequences=True, dropout=0.3))
# model.add(LSTM(lstmoutputdim, dropout=0.3))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

try:
    api_file.loadMode(model, pathh5)
except:
    history = model.fit(trainX, trainY, epochs=20, batch_size=120, validation_split=0.1, verbose=2)
    api_image.showTrainHistory(history, 'loss', 'val_loss')
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score ', trainScore)
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score ', testScore)
    api_file.saveMode(model, pathh5)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
allPredict = model.predict(futureX)
print(allPredict[:10])
pyplot.plot(stock2330_X, stock2330_Y)
pyplot.plot(stock2330_X[look_back:len(trainPredict) + look_back], trainPredict)
pyplot.plot(stock2330_X[len(trainPredict) + look_back * 2 + 1 + 10:len(stock2330_X) - 1], testPredict)
pyplot.show()


# temp = []
# day = 5
# for i in range(day):
#     temp.append(stock2330_Y[len(stock2330_Y) - day + i])
# for i in range(day):
#     data = []
#     data.append(temp[-look_back:])
#     data = numpy.array(data)
#     predict = model.predict(data)
#     print('future stock:', predict)
#     temp.append(predict[0][0])
