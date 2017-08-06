from pandas import DataFrame, concat

from com.kailin.api_image import api_image
from com.kailin.api_file import api_file
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error


def timeseries_to_supervised(data, log=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, log + 1)]
    columns.append(df)
    return concat(columns, axis=1).fillna(0, inplace=True)


dataSource = api_file.readExcel('E:/pythonwork/data/sales-of-shampoo-over-a-three-ye.xlsx')

trainX = dataSource[:24].values
testX = dataSource[24:].values

history = [x for x in trainX]
predictions = list()
for i in range(len(testX)):
    predictions.append(history[-1])
    history.append(testX[i])

print('RMSE : ', sqrt(mean_squared_error(testX, predictions)))

pyplot.plot(testX)
pyplot.plot(predictions)
pyplot.show()
