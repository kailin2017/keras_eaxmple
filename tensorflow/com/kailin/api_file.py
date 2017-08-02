import pandas
from keras.models import Sequential


class api_file:
    def loadMode(model,path):
        model.load_weights(path)

    def saveMode(model,path):
        model.save_weights(path)

    def readExcel(path):
        return pandas.read_excel(path)