import numpy
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from com.kailin.api_image import api_image
from com.kailin.api_file import api_file

numpy.random.seed(10)

(train_image, train_label), (test_image, test_label) = mnist.load_data()

# train_image_hot = train_image.reshape(60000, 784).astype('float32') / 255
# test_image_hot = test_image.reshape(10000, 784).astype('float32') / 255

train_image_hot = train_image.reshape(train_image.shape[0], 28, 28, 1).astype('float32') / 255
test_image_hot = test_image.reshape(test_image.shape[0], 28, 28, 1).astype('float32') / 255

train_label_hot = np_utils.to_categorical(train_label)
test_label_hot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))
# default version
# model.add(Dense(input_dim=784, units=1024, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=512, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# try:
#     # model.load_weights('E:\pythonwork\data\e06.h5')
#     api_file.loadMode(model,'E:\pythonwork\data\e06.h5')
# except:
history = model.fit(x=train_image_hot, y=train_label_hot, validation_split=0.2, epochs=10, batch_size=300, verbose=2)
api_image.showTrainHistory(history, 'acc', 'val_acc')
api_image.showTrainHistory(history, 'loss', 'val_loss')
scores = model.evaluate(test_image_hot, test_label_hot)
print('\n', scores, '\n')

prediction = model.predict_classes(test_image_hot)
api_image.showImageLabelPrediction(test_image, test_label, prediction, 0, 25)
api_image.confusionMatrix(test_label, prediction)

api_file.saveMode(model, 'E:\pythonwork\data\e06.h5')
# model.save_weights('E:\pythonwork\data\e06.h5')
