import numpy
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense

from com.kailin.api_file import api_file
from com.kailin.api_image import api_image

numpy.random.seed(7)


def createDataset(dataset, look_back=1):
    dataX = []
    dataY = []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:i + look_back, 1])
        dataY.append(dataset[i + look_back, 1])
    return numpy.array(dataX), numpy.array(dataY)


dataset = api_file.readExcel(api_file.dataPath + 'international-airline-passengers.xls').values
datasetX = dataset[:, 0]
datasetY = dataset[:, 1]

dataset32 = dataset
train_size = int(len(dataset32) * 0.7)
train, test = dataset32[:train_size], dataset32[train_size:]

print(len(train), len(test))
look_back = 1
trainX, trainY = createDataset(train, look_back)
testX, testY = createDataset(test, look_back)

model = Sequential()
model.add(Dense(units=8, input_dim=look_back, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score ', trainScore)
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score ', testScore)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

pyplot.plot(datasetX, datasetY)
pyplot.plot(datasetX[look_back:len(trainPredict) + look_back], trainPredict)
pyplot.plot(datasetX[len(trainPredict) + look_back * 2 + 1:len(dataset) - 1], testPredict)
pyplot.show()
print('dataset X')
print('train predict \n', trainPredict)
print('test predict \n', testPredict)


