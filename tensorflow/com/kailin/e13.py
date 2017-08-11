import random

import numpy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Flatten, LSTM
from com.kailin.api_e13 import api_e13
from com.kailin.api_image import api_image
from com.kailin.api_file import api_file


dict = {1: '正面', 0: '負面'}
h5path = 'E:/pythonwork/data/e13.h5'
datapath = 'E:/pythonwork/data/aclImdb'
train_text, train_label = api_e13.readSource(datapath, 'train')
test_text, test_label = api_e13.readSource(datapath, 'test')

token = Tokenizer(num_words=4000)
token.fit_on_texts(train_text)
train_text_seq = token.texts_to_sequences(train_text)
test_text_seq = token.texts_to_sequences(test_text)
train_text_seq = sequence.pad_sequences(train_text_seq, maxlen=300)
test_text_seq = sequence.pad_sequences(test_text_seq, maxlen=300)

numpy.random.seed(10)

model = Sequential()
model.add(Embedding(output_dim=32, input_dim=4000, input_length=300))
model.add(Dropout(0.2))
model.add(LSTM(units=32,dropout=0.2))
model.add(Dense(units=4000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
try:
    api_file.loadMode(model, h5path)
except:
    history = model.fit(train_text_seq, train_label, batch_size=125, epochs=40, verbose=2, validation_split=0.2)
    api_image.showTrainHistory(history, 'acc', 'val_acc')
    api_image.showTrainHistory(history, 'loss', 'val_loss')
    scores = model.evaluate(test_text_seq, test_label, verbose=1)
    api_file.saveMode(model, h5path)
    print('\n', scores, '\n')

predict = model.predict_classes(test_text_seq).reshape(-1)


def showSentiment(i):
    print('-------------------------------')
    # print(test_text[i])
    print('No.', i, ' 真實評價:', dict[test_label[i]], ' 預測評價:', dict[predict[i]])


for i in range(1, 20):
    showSentiment(random.randint(1, 25000))
