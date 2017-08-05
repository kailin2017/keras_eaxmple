import numpy
import pandas
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout

from com.kailin.api_file import api_file
from com.kailin.api_image import api_image

numpy.random.seed(10)

def PreprocessDada(df):
    df = df.drop(['name'], axis=1)
    df['age'] = df['age'].fillna(df['age'].mean())
    df['fare'] = df['fare'].fillna(df['fare'].mean())
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    df_hot = pandas.get_dummies(data=df, columns=['embarked']).values
    labels = df_hot[:, 0]
    features = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(df_hot[:, 1:])
    return features, labels


clos = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
Jack = pandas.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.000, 'S'])
Rose = pandas.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])
jrdf = pandas.DataFrame([list(Jack), list(Rose)], columns=clos)
alldf = api_file.readExcel('E:/pythonwork/data/titanic3.xls')
alldf = alldf[clos]
alldf = pandas.concat([alldf, jrdf])

msk = numpy.random.rand(len(alldf)) < 0.8
train_f, train_l = PreprocessDada(alldf[msk])
test_f, test_l = PreprocessDada(alldf[~msk])
all_f, all_l = PreprocessDada(alldf)

model = Sequential()
model.add(Dense(input_dim=9, units=900, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=300, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
try:
    api_file.loadMode(model, 'E:\pythonwork\data\e11.h5')
except:
    history = model.fit(x=train_f, y=train_l, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
    api_image.showTrainHistory(history, 'acc', 'val_acc')
    api_image.showTrainHistory(history, 'loss', 'val_loss')
    scores = model.evaluate(train_f, train_l)
    print('\n', scores, '\n')
    api_file.saveMode(model, 'E:\pythonwork\data\e11.h5')

alldf.insert(len(alldf.columns), 'probability', model.predict(all_f))
print(alldf[-20:])
api_file.saveMode(model, 'E:\pythonwork\data\e11.h5')
