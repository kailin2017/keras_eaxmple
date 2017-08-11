
from keras.models import Sequential
from keras.layers import Dense,  Dropout

from com.kailin.api_image import api_image
from com.kailin.api_file import api_file
from matplotlib import pyplot

import numpy

numpy.random.seed(10)


def timestarp2string(dataset):
    result = []
    for i in range(len(dataset['日期'])):
        result.append(str(dataset['日期'][i])[:10].replace('-', ''))
    dataset['日期'] = result
    return dataset


def createDataset(dataset, look_back=1):
    dataX = []
    dataY = []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 1])
        dataY.append(dataset[i + look_back, 1])
        print(dataset[i:(i + look_back), 1])
    return numpy.array(dataX), numpy.array(dataY)


def futureDataset(data):
    size = len(data)
    dataX = []
    dataX.append(data[size - 2:size - 1])
    return numpy.array(dataX)


stockcode = '0050'
pathXlsx = api_file.dataPath + stockcode + '.xlsx'
pathh5 = api_file.dataPath + stockcode + '.h5'

stock2330 = api_file.readExcel(pathXlsx)
stock2330_X = stock2330['日期'].values
stock2330_Y = stock2330['收盤'].values
stock2330_XY = timestarp2string(stock2330)[['日期', '收盤']].values.astype('float64')

look_back = 5
look_predict = len(stock2330_XY) - 20
train, test = stock2330_XY[:look_predict], stock2330_XY[look_predict:]
trainX, trainY = createDataset(train, look_back)
testX, testY = createDataset(test, look_back)
print(testX)

model = Sequential()
# model.add(LSTM(batch_input_shape=(look_back, len(trainX), look_back),
#                output_dim=look_back * 10,
#                return_sequences=True,
#                stateful=True,
#                dropout=0.3))
model.add(Dense(input_dim=look_back, units=look_back * 10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=8))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

try:
    api_file.loadMode(model, pathh5)
except:
    history = model.fit(trainX, trainY, nb_epoch=1000, batch_size=10, validation_split=0.1, verbose=2)
    api_image.showTrainHistory(history, 'loss', 'val_loss')
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score ', trainScore)
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score ', testScore)
    api_file.saveMode(model, pathh5)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

pyplot.plot(stock2330_X, stock2330_Y)
pyplot.plot(stock2330_X[look_back:len(trainPredict) + look_back], trainPredict)
pyplot.plot(stock2330_X[len(trainPredict) + look_back * 2 + 1:len(stock2330_X) - 1], testPredict)
# pyplot.show()

temp = []
day = 10
for i in range(day):
    temp.append(stock2330_Y[len(stock2330_Y) - day + i])
for i in range(day):
    data = []
    data.append(temp[-look_back:])
    data = numpy.array(data)
    predict = model.predict(data)
    print('future day : ', day, '\n', predict)
    temp.append(predict[0][0])
