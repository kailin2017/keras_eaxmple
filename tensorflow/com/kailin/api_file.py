import pandas
from keras.models import Sequential

class api_file:
    dataPath = "/Users/sinopac/Documents/GitHub/keras_eaxmple/data/"
    def loadMode(model,path):
        model.load_weights(path)

    def saveMode(model,path):
        model.save_weights(path)

    def readExcel(path):
        return pandas.read_excel(path)

    def readCsv(path):
        return pandas.read_csv(path)